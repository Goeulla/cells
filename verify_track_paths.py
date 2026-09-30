"""
verify_track_paths.py -- visual + numeric verification for full-trajectory cell
tracking, built specifically to let a human check tracks by EYE rather than trust
speed numbers blindly. Does not touch, import as executable, or duplicate any
counting/speed-bin logic from cell_speed_binning_exit_v7_clean.py -- only reuses
its detection/matching/core-size building blocks (detect_cells, hungarian_match,
measure_core_size), which are the well-tested primitives, and implements a
separate, much simpler single-pass tracker on top: match existing tracks to new
detections each frame, create new tracks for anything unmatched, expire tracks
that go too long without a match, and record each track's FULL per-frame path
(not just start/end) for both video overlay and numeric inspection.

Why this exists: a real bug was found and fixed in --count_at_line's tracker
(a static, non-moving artifact got absorbed into a real cell's track, corrupting
its speed) by pulling one track's actual frame-by-frame path and looking at it.
This generalizes that exact investigation technique into a reusable, permanent
tool, instead of writing one-off debug scripts each time -- and does it for full
FOV-crossing tracking (not --count_at_line's narrow exit band), since that's the
mode now recommended for measuring a cell's whole visible crossing distance.

Two outputs:
  1. An annotated video with each track's cumulative path drawn as a growing
     colored trail (a distinct color per track ID) plus its current detection
     circled and labeled -- watch it side by side with the raw footage to
     confirm a trail really is one continuously-moving cell, not several
     different objects stitched together.
  2. A per-track summary CSV with objective numeric red flags, so you don't have
     to watch the whole video to find suspicious tracks:
       - straightness = net_displacement / total_path_length (1.0 = perfectly
         straight motion; much less than 1 means real back-and-forth wandering,
         which is expected for real rolling cells but harder to interpret than
         a clean straight crossing)
       - max_single_frame_jump_px -- the single biggest frame-to-frame position
         jump anywhere in the track's life. A large value here, especially
         paired with a long stretch of near-zero movement right after it, is
         exactly the signature of the static-artifact-absorption bug already
         found in --count_at_line: a big unexplained jump onto a different,
         unrelated (and possibly stationary) object.
       - max_stall_frames -- the longest run of consecutive frames where the
         track moved less than --stall_px. A real crossing shouldn't stall for
         long; a big number here means part of this track's life was spent
         sitting on something that wasn't actually moving.

Does NOT implement --count_at_line's exit-boundary/counting/speed-bin logic at
all, does NOT decide what counts as a real crossing, and does NOT write
anything resembling per_object_csv -- it exists purely to let you inspect
tracking quality, independent of and without risk to the counting pipeline.
"""
import argparse
import math
import colorsys
from collections import deque

import cv2
import numpy as np
import pandas as pd

from cell_speed_binning_exit_v7_clean import detect_cells, hungarian_match, measure_core_size


def track_color(tid):
    """Deterministic, visually-distinct BGR color per track ID (golden-angle hue spacing)."""
    hue = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--end_s",   type=float, default=None)
    ap.add_argument("--fps",     type=float, default=None)

    # Detection settings -- same names/defaults as cell_speed_binning_exit_v7_clean.py
    # for the subset detect_cells/measure_core_size actually consume. Kept as CLI args
    # (not hardcoded) so this tool can be pointed at the same settings used for a real
    # analysis run, but framediff/edge-barrier/bridge-break features are intentionally
    # NOT reproduced here -- every real run in this project has left them off, and
    # reproducing their full branching here would mean duplicating a large, easy-to-drift
    # slice of the production preprocessing pipeline for a feature this tool doesn't need.
    ap.add_argument("--mog2_history", type=int, default=500)
    ap.add_argument("--mog2_varThreshold", type=float, default=8)
    ap.add_argument("--mask_thresh", type=int, default=60)
    ap.add_argument("--learning_rate", type=float, default=0.0005)
    ap.add_argument("--gauss_ksize", type=int, default=3)
    ap.add_argument("--open_iter", type=int, default=0)
    ap.add_argument("--close_iter", type=int, default=1)
    ap.add_argument("--erode_iter", type=int, default=0)
    ap.add_argument("--min_area", type=float, default=12)
    ap.add_argument("--max_area", type=float, default=8000)
    ap.add_argument("--min_mean_intensity", type=float, default=0.0)
    ap.add_argument("--min_local_motion", type=float, default=0.0)
    ap.add_argument("--exclude_hole_blobs", action="store_true")
    ap.add_argument("--enable_streak", action="store_true")
    ap.add_argument("--streak_ar", type=float, default=2.2)
    ap.add_argument("--streak_min_len", type=float, default=4)
    ap.add_argument("--streak_min_area", type=float, default=6)
    ap.add_argument("--dead_cell_max_sharpness", type=float, default=None)
    ap.add_argument("--dead_cell_percentile", type=float, default=None)
    ap.add_argument("--dead_cell_sharpness_radius", type=int, default=8)
    ap.add_argument("--use_watershed_split", action="store_true")
    ap.add_argument("--watershed_fg_thresh", type=float, default=0.7)
    ap.add_argument("--watershed_min_split_area", type=float, default=None)
    ap.add_argument("--watershed_min_peak_dist", type=float, default=4.0)
    ap.add_argument("--watershed_prominence_frac", type=float, default=0.25)
    ap.add_argument("--watershed_use_intensity", action="store_true")
    ap.add_argument("--watershed_est_cell_area", type=float, default=None)
    ap.add_argument("--core_size_frac", type=float, default=0.35)

    # This tool's own tracker (independent of, and simpler than, the counting
    # pipeline's -- no exit boundary, no counted/uncounted distinction, just
    # continuous single-pass matching so every track's FULL path can be recorded).
    ap.add_argument("--max_dist", type=float, default=220,
                    help="Matching radius, same meaning and same default as the main "
                         "script's --max_dist -- deliberately NOT tightened here, since "
                         "the point of this tool is to show you what the CURRENT "
                         "production setting actually does, not a hypothetical improved "
                         "one.")
    ap.add_argument("--max_missed", type=int, default=20)
    ap.add_argument("--min_track_len", type=int, default=5,
                    help="Drop tracks shorter than this many frames from BOTH outputs -- "
                         "single-frame noise blips aren't useful to visualize or flag.")
    ap.add_argument("--stall_px", type=float, default=3.0,
                    help="Max px of movement over --stall_window_frames to count as a "
                         "stall for max_stall_frames in the summary CSV.")
    ap.add_argument("--stall_window_frames", type=int, default=20,
                    help="Window size (frames) for the stall-run calculation.")

    ap.add_argument("--m_per_px", type=float, default=None,
                    help="If set, the summary CSV's distance columns are in micrometers "
                         "instead of raw pixels.")

    ap.add_argument("--out_video", type=str, default="track_paths_annotated.mp4",
                    help="Annotated output video with growing colored trails per track.")
    ap.add_argument("--trail_max_len", type=int, default=200,
                    help="Cap on how many past points are drawn per track's trail, so a "
                         "very long-lived track's trail doesn't turn into unreadable "
                         "clutter. Does not affect the recorded path used for the "
                         "summary CSV, only what's drawn.")
    ap.add_argument("--out_summary_csv", type=str, default="track_paths_summary.csv")
    ap.add_argument("--enable_stall_fix", action="store_true",
                    help="Apply the same stall-detection fix now in "
                         "cell_speed_binning_exit_v7_clean.py (--stall_window/--stall_px "
                         "there): drop a track that hasn't moved more than --stall_px in "
                         "--stall_window seconds, instead of letting it keep matching a "
                         "static object forever. Off by default so this tool still shows "
                         "you raw, unfixed tracker behavior when you want to see it -- "
                         "pass this to see the fixed behavior instead, e.g. for a direct "
                         "before/after comparison.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(max(0.0, args.start_s) * fps)
    end_frame_excl = int(args.end_s * fps) if args.end_s else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out_video, fourcc, fps, (W, H))

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=args.mog2_history, varThreshold=args.mog2_varThreshold, detectShadows=False)

    tracks = {}      # tid -> {"cx","cy","missed_count","path":[(frame,cx,cy),...]}
    next_id = 1
    cur_frame = start_frame
    stall_frames = max(1, round(args.stall_window_frames))

    while True:
        if end_frame_excl is not None and cur_frame >= end_frame_excl:
            break
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_sharp = gray
        if args.gauss_ksize > 0:
            k = args.gauss_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        fg = backsub.apply(gray, learningRate=float(args.learning_rate))
        _, th = cv2.threshold(fg, args.mask_thresh, 255, cv2.THRESH_BINARY)
        if args.open_iter > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=args.open_iter)
        if args.close_iter > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=args.close_iter)
        if args.erode_iter > 0:
            th = cv2.erode(th, np.ones((3, 3), np.uint8), iterations=args.erode_iter)

        _, detections, det_boxes = detect_cells(th, frame, gray, args)

        for st in tracks.values():
            st["updated"] = False

        pairs, unmatched = hungarian_match(tracks, detections, args.max_dist)
        for tid, j in pairs:
            cx, cy, *_ = detections[j]
            st = tracks[tid]
            st["cx"], st["cy"] = cx, cy
            st["missed_count"] = 0
            st["updated"] = True
            st["path"].append((cur_frame, cx, cy))
            if args.enable_stall_fix:
                if math.hypot(cx - st["stall_x"], cy - st["stall_y"]) > args.stall_px:
                    st["stall_x"], st["stall_y"], st["stall_frame"] = cx, cy, cur_frame
                elif cur_frame - st["stall_frame"] >= stall_frames:
                    st["stalled"] = True

        for j in unmatched:
            cx, cy, *_ = detections[j]
            tracks[next_id] = dict(cx=cx, cy=cy, missed_count=0, updated=True,
                                    path=[(cur_frame, cx, cy)],
                                    stall_x=cx, stall_y=cy, stall_frame=cur_frame, stalled=False)
            next_id += 1

        if args.enable_stall_fix:
            for tid in [tid for tid, st in tracks.items() if st.get("stalled")]:
                del tracks[tid]

        # Draw current frame with every live track's trail so far.
        vis = frame.copy()
        for tid, st in tracks.items():
            if len(st["path"]) < 2:
                continue
            color = track_color(tid)
            pts = st["path"][-args.trail_max_len:]
            for (_, x0, y0), (_, x1, y1) in zip(pts, pts[1:]):
                cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
            if st["updated"]:
                cx, cy = int(st["cx"]), int(st["cy"])
                cv2.circle(vis, (cx, cy), 5, color, 1)
                cv2.putText(vis, str(tid), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"f={cur_frame} t={cur_frame/fps:.2f}s", (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(vis)

        for tid in list(tracks.keys()):
            st = tracks[tid]
            if not st["updated"]:
                st["missed_count"] += 1
                if st["missed_count"] > args.max_missed:
                    del tracks[tid]

        cur_frame += 1

    cap.release()
    vw.release()
    print(f"Wrote: {args.out_video}")

    # Any track still alive at the end of the loop needs closing out for the summary too.
    finished = list(tracks.values())

    rows = []
    for st in finished:
        path = st["path"]
        if len(path) < args.min_track_len:
            continue
        frames = [p[0] for p in path]
        xs = [p[1] for p in path]
        ys = [p[2] for p in path]
        total_len = sum(math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) for i in range(1, len(path)))
        net_disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        straightness = (net_disp / total_len) if total_len > 0 else float("nan")
        max_jump = max((math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) for i in range(1, len(path))), default=0.0)

        # Longest run where movement over --stall_window_frames stayed within --stall_px.
        max_stall = 0
        anchor_i = 0
        for i in range(1, len(path)):
            d = math.hypot(xs[i] - xs[anchor_i], ys[i] - ys[anchor_i])
            if d > args.stall_px:
                anchor_i = i
            else:
                run = frames[i] - frames[anchor_i]
                max_stall = max(max_stall, run)

        scale = args.m_per_px if args.m_per_px else 1.0
        unit = "um" if args.m_per_px else "px"
        if args.m_per_px:
            scale_um = args.m_per_px * 1e6
        else:
            scale_um = 1.0

        rows.append(dict(
            first_frame=frames[0], last_frame=frames[-1], n_points=len(path),
            total_path_length=total_len * scale_um, net_displacement=net_disp * scale_um,
            straightness=straightness, max_single_frame_jump=max_jump * scale_um,
            max_stall_frames=max_stall, distance_unit=unit,
            start_x=xs[0], start_y=ys[0], end_x=xs[-1], end_y=ys[-1],
        ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_summary_csv, index=False)
    print(f"Wrote: {args.out_summary_csv}  ({len(df)} tracks, "
          f"min_track_len={args.min_track_len} already applied)")
    if len(df):
        suspicious = df[(df["max_single_frame_jump"] > args.max_dist * 0.5) | (df["max_stall_frames"] > args.stall_window_frames * 2)]
        if len(suspicious):
            print(f"\n{len(suspicious)} track(s) flagged as worth a visual check "
                  f"(big single-frame jump and/or a long stall) -- see {args.out_summary_csv}:")
            print(suspicious[["first_frame", "last_frame", "max_single_frame_jump", "max_stall_frames"]].to_string())


if __name__ == "__main__":
    main()
