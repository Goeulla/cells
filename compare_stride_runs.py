"""
compare_stride_runs.py -- aliasing test for predictive_tracker.py.

Run predictive_tracker.py twice on the same video and range, once with
--frame_stride 1 and once with a larger stride (e.g. 2), both with --out_points_csv.
This script matches each stride-N track to the stride-1 track that shares its
detections (same frame, within --match_px), then compares their speeds.

Why this detects aliasing: a track that really follows one cell has the same speed
(um/s) at any stride. A track that instead hops to the next cell in a lane on every
processed frame picks up an extra (cell spacing / stride) per frame, so its speed
changes with stride -- and it usually won't line up with a single stride-1 track at
all. Output is broken down by speed, since aliasing is expected mainly for fast
tracks, where one frame's travel is comparable to the spacing between cells.
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_tracks", required=True, help="stride-1 run: --out_csv")
    ap.add_argument("--base_points", required=True, help="stride-1 run: --out_points_csv")
    ap.add_argument("--test_tracks", required=True, help="stride-N run: --out_csv")
    ap.add_argument("--test_points", required=True, help="stride-N run: --out_points_csv")
    ap.add_argument("--fps", type=float, required=True,
                    help="The fps both tracker runs used (their startup line prints it).")
    ap.add_argument("--um_per_px", type=float, default=0.93)
    ap.add_argument("--match_px", type=float, default=3.0)
    ap.add_argument("--min_shared_frac", type=float, default=0.6,
                    help="Fraction of the stride-N track's points that must coincide with "
                         "one stride-1 track to count as the same cell.")
    ap.add_argument("--bins", default="0,150,300,450,600,800,1200,5000",
                    help="Speed bins (um/s, by stride-1 speed) for the breakdown.")
    ap.add_argument("--out_csv", default="stride_comparison.csv")
    args = ap.parse_args()

    bt = pd.read_csv(args.base_tracks).set_index("track_id")
    tt = pd.read_csv(args.test_tracks).set_index("track_id")
    bp = pd.read_csv(args.base_points)
    tp = pd.read_csv(args.test_points)

    by_frame = defaultdict(list)  # frame -> [(x, y, base_track_id)]
    for r in bp.itertuples(index=False):
        by_frame[r.frame].append((r.x, r.y, r.track_id))
    by_frame = {f: (np.array([[x, y] for x, y, _ in v]), np.array([t for _, _, t in v]))
                for f, v in by_frame.items()}

    rows = []
    for tid, pts in tp.groupby("track_id"):
        votes = Counter()
        for r in pts.itertuples(index=False):
            if r.frame not in by_frame:
                continue
            xy, ids = by_frame[r.frame]
            d = np.hypot(xy[:, 0] - r.x, xy[:, 1] - r.y)
            k = int(np.argmin(d))
            if d[k] <= args.match_px:
                votes[ids[k]] += 1
        match, n_shared = (votes.most_common(1)[0] if votes else (None, 0))
        ok = match is not None and n_shared >= max(3, args.min_shared_frac * len(pts))
        rows.append(dict(test_track=tid, base_track=match if ok else None,
                         test_speed=tt.loc[tid, "speed_um_s"],
                         base_speed=bt.loc[match, "speed_um_s"] if ok else np.nan,
                         shared_points=n_shared, test_points=len(pts)))
    m = pd.DataFrame(rows)
    m.to_csv(args.out_csv, index=False)

    matched = m.dropna(subset=["base_track"])
    matched_base = set(matched.base_track.astype(int))
    edges = [float(x) for x in args.bins.split(",")]
    print(f"stride-N tracks: {len(m)}, matched to a stride-1 track: {len(matched)} "
          f"({100 * len(matched) / max(1, len(m)):.0f}%)")
    print(f"\n{'stride-1 speed bin (um/s)':>26} {'base tracks':>11} {'found at stride N':>18} "
          f"{'median speed ratio':>19} {'ratio off by >20%':>18}")
    for lo, hi in zip(edges, edges[1:]):
        b = bt[(bt.speed_um_s >= lo) & (bt.speed_um_s < hi)]
        mm = matched[(matched.base_speed >= lo) & (matched.base_speed < hi)]
        ratio = mm.test_speed / mm.base_speed
        found = len(set(b.index) & matched_base)
        print(f"{f'{lo:.0f}-{hi:.0f}':>26} {len(b):>11} "
              f"{f'{100 * found / max(1, len(b)):.0f}%':>18} "
              f"{(f'{ratio.median():.2f}' if len(mm) else '-'):>19} "
              f"{(f'{100 * (abs(ratio - 1) > 0.2).mean():.0f}%' if len(mm) else '-'):>18}")
    # Step-level test -- more informative than whole-track matching, which fails
    # whenever the stride-1 track is split into short pieces (common for fast cells).
    # For every stride-N step (frame f -> f+N), find the stride-1 track through the
    # same detection at f and check whether it is at the same place at f+N.
    pos = {(r.track_id, r.frame): (r.x, r.y) for r in bp.itertuples(index=False)}
    to_um_s = args.um_per_px * args.fps
    res = defaultdict(lambda: [0, 0, 0])  # bin -> [agree, disagree, no stride-1 link]
    for tid, pts in tp.groupby("track_id"):
        rows_ = pts.sort_values("frame")[["track_id", "frame", "x", "y"]].to_numpy()
        for (_, f0, x0, y0), (_, f1, x1, y1) in zip(rows_, rows_[1:]):
            v = np.hypot(x1 - x0, y1 - y0) / (f1 - f0) * to_um_s
            k = int(np.searchsorted(edges, v, side="right")) - 1
            if not 0 <= k < len(edges) - 1:
                continue
            if f0 not in by_frame:
                res[k][2] += 1
                continue
            xy, ids = by_frame[f0]
            d = np.hypot(xy[:, 0] - x0, xy[:, 1] - y0)
            i = int(np.argmin(d))
            if d[i] > args.match_px or (ids[i], f1) not in pos:
                res[k][2] += 1
                continue
            xe, ye = pos[(ids[i], f1)]
            res[k][0 if np.hypot(xe - x1, ye - y1) <= args.match_px else 1] += 1
    print("\nStep-level agreement (the stronger test): for each stride-N step, does the "
          "stride-1 track through the same start detection end at the same place?")
    print(f"{'step speed (um/s)':>18} {'agree':>7} {'disagree':>9} {'no stride-1 link':>17} {'agreement':>10}")
    for k in range(len(edges) - 1):
        a, dd, n = res[k]
        print(f"{f'{edges[k]:.0f}-{edges[k + 1]:.0f}':>18} {a:>7} {dd:>9} {n:>17} "
              f"{f'{100 * a / max(1, a + dd):.0f}%':>10}")
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
