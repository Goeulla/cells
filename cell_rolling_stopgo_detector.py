"""
cell_rolling_stopgo_detector.py -- separate tool from cell_speed_binning_exit_v7_clean.py.

Purpose: characterize STOP-AND-GO (rolling/adhesion-interacting) behavior of
individual cells, as distinct from the counting pipeline's job of tallying how
many cells pass a fixed exit line per unit time. A cell that crawls steadily
slow and a cell that repeatedly stops (transient adhesion bond), releases,
and darts forward again can have the SAME average net speed (net displacement
/ total time) -- the counting script's speed metric can't tell them apart.
This tool looks at each cell's FULL frame-by-frame trajectory instead, and
classifies it by whether it shows genuine alternating stop/go episodes.

Reuses the same, already-validated detection pipeline (MOG2 + optional
watershed splitting) from cell_speed_binning_exit_v7_clean.py via direct
import, so detection behavior is identical between the two tools for the
same --mog2_varThreshold/--min_area/etc. Tracking here is intentionally
simpler: whole-frame, single-pass Hungarian matching with one generous
--max_dist (no exit-line/counting concept, no two-pass counted/uncounted
split -- there's nothing analogous to "already counted" here since this
tool's job is to watch a cell's whole visible lifetime, not decide once
whether it crossed a line). This inherits the same chimera-hopping risk at
high density that motivated --count_at_line in the other script -- expect
this tool to be most trustworthy in low/moderate density footage, exactly
where genuine rolling/adhesion behavior is most likely to be visible and
un-confused with neighbors.
"""
import argparse, math, os
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

from cell_speed_binning_exit_v7_clean import detect_cells, hungarian_match


def analyze_trajectory(history, fps, frame_stride, stop_px, min_stop_frames, m_per_px,
                        min_track_span_px=15.0):
    """
    history: list of (frame_idx, cx, cy), one entry per frame the track was
    seen, in order. Returns a dict of stop/go stats + a classification.
    """
    n = len(history)
    if n < 2:
        return dict(n_steps=0, n_stop_episodes=0, stopped_frames=0, moving_frames=0,
                    stopped_s=0.0, moving_s=0.0, mean_moving_speed=0.0,
                    classification="too_short")

    # A track confined to a tiny area over its WHOLE lifetime -- even if per-step
    # jitter occasionally crosses stop_px -- is far more likely static debris/a
    # background artifact than a real cell: even a genuine stop-and-go cell should
    # cover noticeably more ground between its stop and move phases than pure mask/
    # watershed centroid noise on something that never actually moves. Checking
    # this on the bounding-box span of the WHOLE trajectory (not per-step) catches
    # exactly the case per-step noise can slip through: many small jitters that
    # each individually clear stop_px, without the track ever really going anywhere.
    xs = [p[1] for p in history]
    ys = [p[2] for p in history]
    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if span < min_track_span_px:
        return dict(n_steps=n-1, n_stop_episodes=0, stopped_frames=n-1, moving_frames=0,
                    stopped_s=(n-1)*frame_stride/fps, moving_s=0.0, mean_moving_speed=0.0,
                    classification="static_artifact")

    step_period_s = frame_stride / fps  # real seconds between consecutive history entries
    states = []  # True = stopped, False = moving, one per step
    step_speeds = []
    for i in range(1, n):
        _, x0, y0 = history[i-1]
        _, x1, y1 = history[i]
        d = math.hypot(x1 - x0, y1 - y0)
        states.append(d <= stop_px)
        step_speeds.append(d / step_period_s)

    # Group into runs of consecutive identical state.
    runs = []  # (is_stopped, length)
    cur_state, cur_len = states[0], 1
    for s in states[1:]:
        if s == cur_state:
            cur_len += 1
        else:
            runs.append((cur_state, cur_len))
            cur_state, cur_len = s, 1
    runs.append((cur_state, cur_len))

    # Only count a "stopped" run as a real stop episode if it lasts at least
    # min_stop_frames steps -- a single-frame dip below stop_px is more likely
    # detection jitter than a genuine adhesion pause.
    n_stop_episodes = sum(1 for is_stopped, length in runs
                          if is_stopped and length >= min_stop_frames)
    stopped_frames = sum(length for is_stopped, length in runs if is_stopped)
    moving_frames  = sum(length for is_stopped, length in runs if not is_stopped)
    moving_speeds  = [sp for sp, s in zip(step_speeds, states) if not s]
    mean_moving_speed = float(np.mean(moving_speeds)) if moving_speeds else 0.0
    if m_per_px:
        mean_moving_speed *= m_per_px

    if n_stop_episodes == 0:
        classification = "free_flowing"
    elif moving_frames == 0:
        classification = "stationary"
    else:
        classification = "rolling"

    return dict(
        n_steps=len(states), n_stop_episodes=n_stop_episodes,
        stopped_frames=stopped_frames, moving_frames=moving_frames,
        stopped_s=stopped_frames * step_period_s, moving_s=moving_frames * step_period_s,
        mean_moving_speed=mean_moving_speed, classification=classification)


def render_rolling_highlight(video_path, rolling_histories, fps, frame_stride,
                              warmup_start, out_path, stop_px, trail_len=30):
    """
    Second pass, only run after all tracks are finalized and we know which ones
    ended up classified "rolling". Re-reads the video and draws ONLY those
    tracks, each with a fading trail of its last --trail_len positions, instead
    of every currently-active track (which was the actual usability problem --
    with 50-100+ IDs on screen at once and changing every frame, following one
    specific track by eye was nearly impossible even when the tracking was
    correct). Colors: green=moving this step, red=stopped this step, same as
    the main debug video, so the two are visually consistent.
    """
    if not rolling_histories:
        print("No rolling tracks to highlight -- skipping highlight video.")
        return

    # frame_idx -> list of (tid, cx, cy, is_stopped) to draw this frame, precomputed
    # from each track's own history so the render loop is just a lookup.
    per_frame = defaultdict(list)
    for tid, hist in rolling_histories.items():
        for i, (f, cx, cy) in enumerate(hist):
            is_stopped = False
            if i > 0:
                _, x0, y0 = hist[i-1]
                is_stopped = math.hypot(cx-x0, cy-y0) <= stop_px
            per_frame[f].append((tid, cx, cy, is_stopped))

    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return
    H, W = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (2*W, H))

    trails = defaultdict(list)  # tid -> list of recent (cx,cy,is_stopped)
    last_frame_of = {tid: hist[-1][0] for tid, hist in rolling_histories.items()}
    min_f = min(per_frame) if per_frame else 0
    max_f = max(per_frame) if per_frame else 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    abs_frame = warmup_start
    while True:
        if abs_frame > max_f:
            break
        if frame_stride > 1 and (abs_frame - warmup_start) % frame_stride != 0:
            if not cap.grab():
                break
            abs_frame += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break
        cur = abs_frame
        abs_frame += 1
        if cur < min_f:
            continue

        for tid, cx, cy, is_stopped in per_frame.get(cur, []):
            trails[tid].append((cx, cy, is_stopped))
            if len(trails[tid]) > trail_len:
                trails[tid].pop(0)

        orig = frame.copy()
        annotated = frame.copy()
        for tid, pts in trails.items():
            if not pts:
                continue
            for i in range(1, len(pts)):
                x0,y0,_ = pts[i-1]; x1,y1,s1 = pts[i]
                col = (0,0,255) if s1 else (0,255,0)
                cv2.line(annotated, (x0,y0), (x1,y1), col, 1)
            lx, ly, ls = pts[-1]
            col = (0,0,255) if ls else (0,255,0)
            cv2.circle(annotated, (lx,ly), 4, col, -1)
            cv2.putText(annotated, str(tid), (lx+5, ly-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(annotated, f"t={cur/fps:.1f}s rolling_tracks_visible={len(trails)}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(orig, "original", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        vw.write(np.hstack([orig, annotated]))

        # Drop a track's trail once its own history is finished (after drawing
        # its final point this frame), so it disappears instead of lingering
        # frozen on screen for the rest of the video.
        for tid in [t for t in trails if cur >= last_frame_of.get(t, -1)]:
            del trails[tid]

    cap.release()
    vw.release()
    print(f"Wrote: {out_path}  ({len(rolling_histories)} rolling tracks highlighted)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--end_s", type=float, default=None)
    ap.add_argument("--warmup_s", type=float, default=45.0,
                    help="Seconds of real prior footage to feed MOG2 before --start_s, so the "
                         "background model is settled before analysis begins. Do NOT set this "
                         "to 0 to save time when testing -- confirmed directly: with no warmup, "
                         "an untrained MOG2 model flags noisy/unstable blobs indiscriminately "
                         "for the first several seconds, and a track that starts during that "
                         "window can show spurious jittery 'stopped' steps that look like real "
                         "stop-and-go behavior but are actually just detector noise. On a real "
                         "non-functionalized (negative control) clip, skipping warmup produced "
                         "78%% false 'rolling' classifications; with proper warmup on the same "
                         "video, that dropped to 0%% (all correctly free_flowing). This needs "
                         "real footage before --start_s to work -- if testing an isolated short "
                         "clip, set --start_s well after the clip's own beginning so there's "
                         "actual prior video for warmup to use, not just silence/nothing.")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Frame-by-frame trajectory analysis is sensitive to sampling gaps "
                         "the same way the counting script's --count_at_line is -- default is "
                         "1 (every frame) here, not 5, since a coarse stride can make a genuine "
                         "brief stop invisible or turn continuous motion into apparent jumps.")

    # Detection pipeline -- same names/defaults as cell_speed_binning_exit_v7_clean.py so
    # settings validated there (e.g. --mog2_varThreshold 8) transfer directly.
    ap.add_argument("--mog2_history", type=int, default=500)
    ap.add_argument("--mog2_varThreshold", type=float, default=16)
    ap.add_argument("--learning_rate", type=float, default=0.0005)
    ap.add_argument("--mask_thresh", type=int, default=60)
    ap.add_argument("--gauss_ksize", type=int, default=3)
    ap.add_argument("--min_area", type=float, default=12)
    ap.add_argument("--max_area", type=float, default=8000)
    ap.add_argument("--use_watershed_split", action="store_true")
    ap.add_argument("--watershed_use_intensity", action="store_true")
    ap.add_argument("--watershed_min_peak_dist", type=float, default=None)
    ap.add_argument("--watershed_prominence_frac", type=float, default=0.25)
    ap.add_argument("--watershed_fg_thresh", type=float, default=0.7)

    # Tracking.
    ap.add_argument("--max_dist", type=float, default=500,
                    help="Generous whole-frame matching radius, same role as in the counting "
                         "script's uncounted-track pass. No tight/counted second pass here -- "
                         "this tool has no 'already resolved' concept, it watches full "
                         "lifetimes.")
    ap.add_argument("--max_missed", type=int, default=5,
                    help="Frames a track can go unmatched before being finalized. Kept modest "
                         "(not 0) since a genuinely stopped cell may not be re-detected as "
                         "cleanly frame to frame even while it's still there -- but a long gap "
                         "combined with a stale, frozen position prediction is exactly the "
                         "condition that let unrelated tracks get merged via reacquisition (see "
                         "the per-track reacquisition-distance cap in the matching loop, which "
                         "is the main defense against that; this bound is a second, coarser "
                         "backstop so a track can't sit in limbo indefinitely).")
    ap.add_argument("--min_seen_count", type=int, default=5,
                    help="Minimum trajectory length (frames) for a track to be analyzed at "
                         "all -- short tracks don't have enough steps for a stop/go pattern to "
                         "mean anything.")

    # Stop/go classification.
    ap.add_argument("--stop_px", type=float, default=3.0,
                    help="Max per-frame displacement (px) to count a step as 'stopped'. "
                         "Chosen as a fixed pixel distance (not a speed) so it doesn't need "
                         "re-tuning across videos with different fps/frame_stride.")
    ap.add_argument("--min_stop_frames", type=int, default=3,
                    help="Minimum consecutive 'stopped' steps to count as a real stop episode, "
                         "not single-frame jitter.")
    ap.add_argument("--min_track_span_px", type=float, default=15.0,
                    help="A track whose ENTIRE trajectory (not just one step) stays within "
                         "this many px of its own bounding box is classified 'static_artifact' "
                         "instead of rolling/stationary -- catches background debris whose "
                         "per-step mask/watershed centroid noise occasionally crosses --stop_px "
                         "without the track ever really going anywhere.")
    ap.add_argument("--m_per_px", type=float, default=None)

    ap.add_argument("--out_csv", type=str, default="rolling_stopgo.csv")
    ap.add_argument("--debug_video", type=str, default=None)
    ap.add_argument("--draw_detections", action="store_true")
    ap.add_argument("--debug_video_rolling_only", type=str, default=None,
                    help="Second-pass debug video showing ONLY the tracks that ended up "
                         "classified 'rolling', each with a fading trail of its own path, "
                         "instead of every track on screen at once (with 50-100+ simultaneous "
                         "IDs, following one specific track by eye in the regular --debug_video "
                         "is nearly impossible even when the tracking itself is correct).")
    args = ap.parse_args()

    if args.use_watershed_split:
        try:
            import skimage  # noqa: F401
        except ImportError:
            raise SystemExit("--use_watershed_split requires scikit-image.")

    # detect_cells() also reads these -- not exposed as CLI flags here since this tool
    # isn't about dead-cell/streak classification, just fixed off/neutral.
    for name, val in [("watershed_min_split_area", None), ("watershed_est_cell_area", None),
                       ("exclude_hole_blobs", False), ("min_mean_intensity", 0.0),
                       ("min_local_motion", 0.0), ("enable_streak", False),
                       ("streak_min_area", 6), ("streak_min_len", 4), ("streak_ar", 2.2),
                       ("dead_cell_max_sharpness", None), ("dead_cell_percentile", None),
                       ("dead_cell_sharpness_radius", 8),
                       ("use_framediff", False), ("open_iter", 0), ("close_iter", 1),
                       ("bridge_break_size", 0), ("erode_iter", 0), ("use_edge_barrier", False)]:
        setattr(args, name, val)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_s = max(0.0, args.start_s)
    start_frame = int(start_s * fps)
    end_frame_excl = int(args.end_s * fps) if args.end_s else None
    warmup_frames = int(max(0.0, args.warmup_s) * fps)
    warmup_start = max(0, start_frame - warmup_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    ret, frame0 = cap.read()
    if not ret:
        raise SystemExit("Cannot read first frame.")
    H, W = frame0.shape[:2]

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=args.mog2_history, varThreshold=args.mog2_varThreshold, detectShadows=False)

    next_id = 1
    tracks = {}
    finished_rows = []

    vw = None
    if args.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.debug_video, fourcc, fps, (2 * W, H))

    rolling_histories = {}  # track_id -> history, only for classification=="rolling"

    def finalize(tid, st):
        stats = analyze_trajectory(st["history"], fps, args.frame_stride,
                                    args.stop_px, args.min_stop_frames, args.m_per_px,
                                    args.min_track_span_px)
        if len(st["history"]) < args.min_seen_count:
            return
        first_frame, sx, sy = st["history"][0]
        last_frame, ex, ey = st["history"][-1]
        finished_rows.append(dict(
            track_id=tid, first_frame=first_frame, last_frame=last_frame,
            seen_count=len(st["history"]), duration_s=(last_frame-first_frame)/fps,
            start_x=sx, start_y=sy, end_x=ex, end_y=ey, **stats))
        if stats["classification"] == "rolling":
            rolling_histories[tid] = list(st["history"])

    prev_gray = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    abs_frame = warmup_start

    while True:
        if end_frame_excl is not None and abs_frame >= end_frame_excl:
            break
        if args.frame_stride > 1 and (abs_frame - warmup_start) % args.frame_stride != 0:
            if not cap.grab():
                break
            abs_frame += 1
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        cur_frame = abs_frame
        abs_frame += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.gauss_ksize > 0:
            k = args.gauss_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        fg = backsub.apply(gray, learningRate=float(args.learning_rate))
        _, th = cv2.threshold(fg, args.mask_thresh, 255, cv2.THRESH_BINARY)
        if args.close_iter > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8),
                                  iterations=args.close_iter)
        prev_gray = gray

        if cur_frame < start_frame:
            continue

        contours, detections, det_boxes = detect_cells(th, frame, gray, args)

        for tid in tracks:
            tracks[tid]["updated"] = False

        # Match against each track's PREDICTED next position (linearly extrapolated
        # from its own last real step), not its last known position. Confirmed
        # necessary: matching on raw last-position can't tell "this cell is still
        # here" from "a different, faster cell just passed through here" -- a
        # genuinely stopped cell's own detection can get stolen by a passing
        # free-flowing cell that wanders within max_dist of where the stopped cell
        # used to be, creating a fake stop-then-jump-then-stop pattern that isn't
        # real. A track with near-zero recent velocity predicts staying near-zero,
        # so a fast passer (far from that prediction) won't win the match just for
        # being spatially close. Tradeoff: right at a genuine stop->release moment,
        # the prediction (still based on the stopped period) can undershoot the
        # real jump enough to miss the match for one frame, splitting that cell
        # into two track IDs instead of one continuous "rolling" track -- less
        # harmful than the false-positive this fixes, but worth knowing about.
        tracks_for_matching = {}
        for tid, st in tracks.items():
            hist = st["history"]
            if len(hist) >= 2:
                _, x0, y0 = hist[-2]
                _, x1, y1 = hist[-1]
                pred_cx, pred_cy = x1 + (x1 - x0), y1 + (y1 - y0)
            else:
                pred_cx, pred_cy = st["cx"], st["cy"]
            tracks_for_matching[tid] = dict(st, cx=pred_cx, cy=pred_cy)

        pairs, unmatched = hungarian_match(tracks_for_matching, detections, args.max_dist)

        # Post-filter: max_dist=500 is a deliberately generous whole-frame radius
        # meant for fast passing cells, but it's dangerously generous for a track
        # that has been missed for a few frames -- during a gap, hungarian_match
        # will happily reacquire ANY detection within 500px of the frozen
        # prediction, even one that belongs to a completely different, unrelated
        # cell that just wandered into range. This created "teleporting" trails
        # confirmed visually (t=21.1s screenshot: trails spanning most of the
        # frame that don't correspond to any real moving cell). Fix: cap the
        # allowed reacquisition jump to what THIS track's own trajectory has
        # actually shown it capable of, scaled by how long it's been missing (a
        # genuinely fast track can legitimately cover more ground over a longer
        # gap; a track that's been sitting still has no business grabbing a
        # detection far away just because the global radius allows it).
        confirmed_pairs = []
        rejected_det_idxs = []
        for tid, j in pairs:
            st = tracks[tid]
            hist = st["history"]
            if len(hist) >= 2:
                steps = [math.hypot(hist[k][1]-hist[k-1][1], hist[k][2]-hist[k-1][2])
                        for k in range(1, len(hist))]
                own_max_step = max(steps)
            else:
                own_max_step = args.max_dist  # no track record yet -- don't restrict first real match
            gap = st["missed_count"] + 1
            reacquire_cap = max(args.stop_px * 4.0, own_max_step * 1.5) * gap
            pred_cx, pred_cy = tracks_for_matching[tid]["cx"], tracks_for_matching[tid]["cy"]
            det_cx, det_cy, _is_streak, _is_dead = detections[j]
            jump = math.hypot(det_cx - pred_cx, det_cy - pred_cy)
            if jump <= reacquire_cap:
                confirmed_pairs.append((tid, j))
            else:
                rejected_det_idxs.append(j)
        pairs = confirmed_pairs
        unmatched = list(unmatched) + rejected_det_idxs

        for tid, j in pairs:
            cx, cy, _is_streak, _is_dead = detections[j]
            st = tracks[tid]
            st["cx"], st["cy"] = cx, cy
            st["missed_count"] = 0
            st["updated"] = True
            st["history"].append((cur_frame, cx, cy))

        for j in unmatched:
            cx, cy, _is_streak, _is_dead = detections[j]
            tracks[next_id] = dict(cx=cx, cy=cy, missed_count=0, updated=True,
                                   history=[(cur_frame, cx, cy)])
            next_id += 1

        for tid, st in list(tracks.items()):
            if not st["updated"]:
                st["missed_count"] += 1
                if st["missed_count"] > args.max_missed:
                    finalize(tid, st)
                    tracks.pop(tid, None)

        if vw is not None:
            orig = frame.copy()
            annotated = frame.copy()
            if args.draw_detections:
                for (x,y,w,h,is_s,is_dead) in det_boxes:
                    cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,0,255),1)
            for tid, st in tracks.items():
                hist = st["history"]
                if len(hist) >= 2:
                    d = math.hypot(hist[-1][1]-hist[-2][1], hist[-1][2]-hist[-2][2])
                    col = (0,0,255) if d <= args.stop_px else (0,255,0)  # red=stopped, green=moving
                else:
                    col = (255,255,0)
                cv2.circle(annotated, (st["cx"], st["cy"]), 3, col, -1)
                cv2.putText(annotated, str(tid), (st["cx"]+4, st["cy"]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(annotated, f"t={cur_frame/fps:.1f}s trk={len(tracks)}",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(orig, "original", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            vw.write(np.hstack([orig, annotated]))

    for tid, st in list(tracks.items()):
        finalize(tid, st)

    cap.release()
    if vw: vw.release()

    df = pd.DataFrame(finished_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}  ({len(df)} tracks analyzed)")
    if len(df):
        print(df["classification"].value_counts().to_string())

    if args.debug_video_rolling_only:
        render_rolling_highlight(args.video, rolling_histories, fps, args.frame_stride,
                                 warmup_start, args.debug_video_rolling_only, args.stop_px)


if __name__ == "__main__":
    main()
