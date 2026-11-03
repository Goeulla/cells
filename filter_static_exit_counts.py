"""
filter_static_exit_counts.py -- remove exit-line counts of cells that were not moving.

Standalone post-filter for cell_speed_binning_exit_v7_clean.py (which it does not
import or change). Run the counting script with --per_object_csv, then run this on
the same video with that CSV.

The problem it fixes. The counting script counts a detection as a crossing as soon
as it is within --exit_margin_px of the exit edge. A cell that is stuck at the edge
is still detected (the background model absorbs it only very slowly), and its
detection flickers on and off. Each time it comes back it gets a new track and is
counted again. With no flow at all, 3 minutes of the F1 recording gave 41 counts
at --frame_stride 5 and 91 at stride 1.

The test. A cell that really crossed the line was somewhere else --lag_s earlier.
For each counted object, its exit position at the counting frame is compared with
the same place --lag_s and --lag_s/2 earlier, allowing --search_px of drift, in two
ways:
  - similarity of a (2 x --patch_px + 1) patch (normalised cross-correlation), and
  - mean grey-level change of a small core (radius --core_px) at the cell itself.
The count is rejected as stationary if, at either lag, the patch is nearly the same
(NCC >= --ncc_thr) or the core barely changed (< --core_diff_max). "Either lag":
a stuck cell's halo can change shape within a second (e.g. slight focus change),
while a cell that really flowed in is absent at both times. The core test catches
cells right at the frame edge, where the patch is mostly background above them.
Counts in the first --lag_s of the video, where there is no earlier frame, are
checked against frames AFTER the count instead: a crossing cell has left by then.
A cell that arrives and then stops at the line is still counted once.
Slowest speed still counted: roughly one cell radius per --lag_s (~5 um/s at 1 s).

Tested on F1 (last 3 min, no flow) plus a copy with 20 synthetic cells pasted in,
moving down at 20-600 um/s: all 87 stationary counts were rejected (stride 1), and
every synthetic crossing the counting script found was kept. Thresholds were set
on this one recording -- check the --sheet_out thumbnails on other videos.

Usage
  python3 cell_speed_binning_exit_v7_clean.py --video V.mp4 ... --per_object_csv obj.csv
  python3 filter_static_exit_counts.py --video V.mp4 --per_object_csv obj.csv \\
      --speed_bins 0,50,100,200,400,800,5000 --bin_seconds 10

Use the same --fps / --true_duration_min / --duration_inflation_ratio as the counting
run, so exit times map back to the same frames.

Outputs: <per_object_csv>_checked.csv (all objects, with ncc_* and
rejected_stationary columns), <per_object_csv>_kept_per_time.csv (per-time counts
after filtering, next to the original totals), and optionally --sheet_out, a
thumbnail sheet of rejected and kept objects for checking by eye.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd


def best_ncc(cur, ref, x, y, r, s):
    """Max NCC of the (2r+1) patch of `cur` at (x, y) against `ref`, searching +-s px."""
    H, W = cur.shape
    x0, x1 = max(s, x - r), min(W - s, x + r + 1)
    y0, y1 = max(s, y - r), min(H - s, y + r + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return float("nan")
    tpl = cur[y0:y1, x0:x1]
    if tpl.std() < 1e-3:
        return float("nan")
    reg = ref[y0 - s:y1 + s, x0 - s:x1 + s]
    return float(cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED).max())


def core_diff(cur, ref, x, y, r, s):
    """Smallest mean |cur - ref| over a (2r+1) core at (x, y), allowing +-s px of drift."""
    H, W = cur.shape
    y0, y1, x0, x1 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
    a = cur[y0:y1, x0:x1]
    best = float("nan")
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            if y0 + dy < 0 or y1 + dy > H or x0 + dx < 0 or x1 + dx > W:
                continue
            d = float(np.abs(a - ref[y0 + dy:y1 + dy, x0 + dx:x1 + dx]).mean())
            best = d if not best <= d else best
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--per_object_csv", required=True, help="--per_object_csv of the counting run")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--true_duration_min", type=float, default=None)
    ap.add_argument("--duration_inflation_ratio", type=float, default=1.132)
    ap.add_argument("--lag_s", type=float, default=1.0,
                    help="How far back to look. Cells slower than ~one radius per lag_s are "
                         "treated as stationary.")
    ap.add_argument("--ncc_thr", type=float, default=0.85,
                    help="Patch similarity at or above which the spot is judged unchanged.")
    ap.add_argument("--core_diff_max", type=float, default=8.0,
                    help="Core grey-level change below which the spot is judged unchanged. "
                         "In grey levels of the video; on F1, stuck cells gave ~0-3 (up to "
                         "~20 when the halo changed shape) and moving cells 31-74.")
    ap.add_argument("--core_px", type=int, default=4, help="Core radius (px) for --core_diff_max.")
    ap.add_argument("--patch_px", type=int, default=10, help="Patch half-size (px).")
    ap.add_argument("--search_px", type=int, default=2, help="Allowed drift (px) between frames.")
    ap.add_argument("--speed_bins", default=None, help="Same as the counting run, to rebuild speed-bin columns.")
    ap.add_argument("--bin_seconds", type=float, default=1.0)
    ap.add_argument("--sheet_out", default=None, help="Optional PNG of thumbnails (red = rejected).")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.fps:
        fps = args.fps
    elif args.true_duration_min:
        fps = n_frames / (args.true_duration_min * 60.0)
    else:
        fps = file_fps * args.duration_inflation_ratio
    lag = max(1, int(round(args.lag_s * fps)))
    half = max(1, lag // 2)

    obj = pd.read_csv(args.per_object_csv)
    obj["exit_frame"] = np.round(obj.time_exit_s_abs * fps).astype(int)
    need = set()
    for f in obj.exit_frame:
        need.update([f, f - half, f - lag, f + half, f + lag])
    need = {f for f in need if f >= 0}

    frames, idx = {}, 0
    last = max(need) if need else -1
    while idx <= last:
        if idx in need:
            ok, fr = cap.read()
            if not ok:
                break
            frames[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        elif not cap.grab():
            break
        idx += 1
    cap.release()

    n1, n2, cdl, rej, note = [], [], [], [], []
    for r in obj.itertuples():
        f, x, y = r.exit_frame, int(r.x_exit), int(r.y_exit)
        if f not in frames:
            n1.append(np.nan); n2.append(np.nan); cdl.append(np.nan); rej.append(0); note.append("frame_not_read")
            continue
        sign = -1 if f - lag in frames else 1          # look back; forward near the start
        if f + sign * lag not in frames or f + sign * half not in frames:
            n1.append(np.nan); n2.append(np.nan); cdl.append(np.nan); rej.append(0); note.append("not_checkable")
            continue
        a = best_ncc(frames[f], frames[f + sign * lag], x, y, args.patch_px, args.search_px)
        b = best_ncc(frames[f], frames[f + sign * half], x, y, args.patch_px, args.search_px)
        cd = min(core_diff(frames[f], frames[f + sign * lag], x, y, args.core_px, args.search_px),
                 core_diff(frames[f], frames[f + sign * half], x, y, args.core_px, args.search_px))
        n1.append(a); n2.append(b); cdl.append(cd)
        stat = max(a, b) >= args.ncc_thr or cd < args.core_diff_max
        rej.append(int(stat))
        note.append(("stationary" if stat else "moved") + ("" if sign < 0 else "_checked_forward"))
    obj["ncc_lag"], obj["ncc_half_lag"], obj["core_diff"] = n1, n2, cdl
    obj["rejected_stationary"], obj["check"] = rej, note

    stem = os.path.splitext(args.per_object_csv)[0]
    obj.to_csv(f"{stem}_checked.csv", index=False)

    start_s = float((obj.time_exit_s_abs - obj.time_exit_s_rel).median()) if len(obj) else 0.0
    obj["tb"] = (obj.time_exit_s_rel // args.bin_seconds).astype(int)
    kept = obj[obj.rejected_stationary == 0]
    nb = int(obj.tb.max()) + 1 if len(obj) else 0
    table = pd.DataFrame({"tb": range(nb)})
    table["time_start_s"] = start_s + table.tb * args.bin_seconds
    table["time_end_s"] = table.time_start_s + args.bin_seconds
    table["counted_original"] = table.tb.map(obj.groupby("tb").size()).fillna(0).astype(int)
    table["rejected_stationary"] = table.tb.map(obj.groupby("tb").rejected_stationary.sum()).fillna(0).astype(int)
    table["counted_kept"] = table.tb.map(kept.groupby("tb").size()).fillna(0).astype(int)
    if args.speed_bins:
        edges = [float(v) for v in args.speed_bins.split(",")]
        sp = kept[kept.counted_as == "speedbin"]
        for lo, hi in zip(edges, edges[1:]):
            s = sp[(sp.speed >= lo) & (sp.speed < hi)]
            table[f"kept_speedbin_{lo:g}_{hi:g}"] = table.tb.map(s.groupby("tb").size()).fillna(0).astype(int)
    table.drop(columns="tb").to_csv(f"{stem}_kept_per_time.csv", index=False)

    if args.sheet_out and len(obj):
        thumbs = []
        cap = cv2.VideoCapture(args.video)
        for r in obj.sort_values("rejected_stationary").itertuples():
            if r.exit_frame not in frames:
                continue
            g = frames[r.exit_frame]
            x, y, c = int(r.x_exit), int(r.y_exit), 24
            pad = cv2.copyMakeBorder(g, c, c, c, c, cv2.BORDER_CONSTANT, value=0)
            t = cv2.cvtColor(pad[y:y + 2 * c, x:x + 2 * c].astype(np.uint8), cv2.COLOR_GRAY2BGR)
            t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            col = (0, 0, 255) if r.rejected_stationary else (0, 200, 0)
            cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), col, 2)
            cv2.putText(t, f"{r.time_exit_s_abs:.1f}s", (3, 12), 0, 0.35, (255, 255, 255), 1)
            thumbs.append(t)
        cols = 12
        while len(thumbs) % cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        sheet = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
        cv2.imwrite(args.sheet_out, sheet)

    n = len(obj)
    print(f"fps={fps:.3f}, lag={lag} frames ({lag / fps:.2f} s)")
    print(f"counted by the counting script: {n}")
    print(f"rejected as stationary (already there up to {args.lag_s:g} s earlier): "
          f"{int(obj.rejected_stationary.sum())}")
    print(f"kept: {n - int(obj.rejected_stationary.sum())}"
          + (f"  (incl. {int(obj.check.isin(['frame_not_read', 'not_checkable']).sum())} not checkable)"
             if obj.check.isin(['frame_not_read', 'not_checkable']).any() else ""))
    print(f"Wrote: {stem}_checked.csv, {stem}_kept_per_time.csv"
          + (f", {args.sheet_out}" if args.sheet_out else ""))


if __name__ == "__main__":
    main()
