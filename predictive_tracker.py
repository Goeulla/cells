"""
predictive_tracker.py -- flow-direction-aware, velocity-predictive cell tracker.

Standalone: imports only detect_cells from cell_speed_binning_exit_v7_clean.py (the
detection step) and does not modify or reuse either counting script's tracking.

Why a third tracker. On dense footage (5_minute.mp4) we measured, without any
tracking, that cells move ~12 px/frame typically (~230 um/s) and up to 20-40
px/frame, while detection boxes are only ~9 px tall. That breaks both earlier
approaches:
  - nearest-centroid matching (the counting scripts) has to search a radius larger
    than the spacing between cells, so tracks hop between neighbours (about a third
    of all steps went backward against the flow);
  - box-overlap matching (TrafficFlowAnalysis, the earlier paper's tool) needs a cell
    to move less than its own size per frame, so it only ever follows slow cells
    (median 78 um/s) and fragments fast ones.

This tracker combines the paper's forward-movement rule with a per-track velocity
prediction -- essentially the "look-ahead window" the paper describes but whose
code path is never called:
  - every track predicts where it will be next from its own recent velocity, and a
    detection can only continue it if it lands close to that prediction (tight
    along-flow and across-flow gates, scaled with the track's speed);
  - a brand-new track (one point, velocity unknown) may only continue forward along
    the flow, within a narrow lane, and prefers the step closest to a population
    speed prior (seeded from a tracking-free cross-correlation of the video, then
    updated from confirmed tracks);
  - a wrong first pairing is self-correcting: the third point must match the
    prediction implied by the first two, so a mis-paired track dies instead of
    being confirmed;
  - assignment is one-to-one (Hungarian), established tracks first.
Objects moving less than --min_step_px per frame (stuck cells, debris) can never
extend a track, so static artifacts cannot be absorbed at all.

Outputs a per-track CSV (speed = least-squares slope of along-flow position vs time,
not start-to-end over elapsed time) and prints a validation block comparing the
tracks' step speeds against the tracking-free cross-correlation reference.
"""
import argparse
import math
import types
from collections import deque

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from cell_speed_binning_exit_v7_clean import detect_cells, measure_core_size, solve_height_wall_corrected

FLOW_VEC = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}

# detect_cells reads these; defaults match cell_speed_binning_exit_v7_clean.py
DETECT_DEFAULTS = dict(
    min_area=12, max_area=8000, min_mean_intensity=0.0, min_local_motion=0.0,
    exclude_hole_blobs=False, enable_streak=False, streak_ar=2.2, streak_min_len=4,
    streak_min_area=6, dead_cell_max_sharpness=None, dead_cell_percentile=None,
    dead_cell_sharpness_radius=8, use_watershed_split=False, watershed_fg_thresh=0.7,
    watershed_min_split_area=None, watershed_min_peak_dist=4.0,
    watershed_prominence_frac=0.25, watershed_use_intensity=False,
    watershed_est_cell_area=None, core_size_frac=0.35)


def to_flow(x, y, flow_dir):
    """(x, y) image coords -> (along, across) flow coords."""
    fx, fy = FLOW_VEC[flow_dir]
    return x * fx + y * fy, -x * fy + y * fx


def xcorr_speed_spectrum(video, start_frame, n_frames, flow_dir, max_shift, col_step=4):
    """
    Tracking-free speed estimate: correlate each frame's moving foreground with the
    next frame's, shifted along the flow axis. Returns (shifts_px, normalized score).
    The peak is the dominant per-frame displacement of moving material.
    """
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    G = []
    for _ in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if flow_dir in ("right", "left"):
            g = g.T
        if flow_dir in ("up", "left"):
            g = g[::-1]
        G.append(g)
    cap.release()
    if len(G) < 3:
        return np.array([0]), np.array([1.0])
    G = np.stack(G)
    T, H, W = G.shape
    F = np.empty_like(G)
    for s in range(0, T, 120):
        blk = G[s:s + 120]
        F[s:s + 120] = np.abs(blk - np.median(blk, axis=0))
    F[F < 8] = 0
    F = F[:, :, ::col_step]
    shifts = np.arange(-10, max_shift + 1)
    score = np.zeros(len(shifts))
    for t in range(T - 1):
        a, b = F[t], F[t + 1]
        for k, s in enumerate(shifts):
            if s >= 0:
                score[k] += (a[:H - s] * b[s:]).sum()
            else:
                score[k] += (a[-s:] * b[:H + s]).sum()
    return shifts, score / score.max()


def assign(cost, gate):
    """One-to-one assignment minimizing total cost; pairs above gate are rejected."""
    if cost.size == 0:
        return []
    big = gate * 1e3
    c = np.where(cost <= gate, cost, big)
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(c)
        return [(r, k) for r, k in zip(rows, cols) if c[r, k] <= gate]
    pairs, used_r, used_c = [], set(), set()
    for idx in np.argsort(c, axis=None):
        r, k = divmod(int(idx), c.shape[1])
        if c[r, k] > gate:
            break
        if r in used_r or k in used_c:
            continue
        pairs.append((r, k)); used_r.add(r); used_c.add(k)
    return pairs


def track_velocity(pts, k):
    """Least-squares (v_along, v_across) in px/frame over the last k points."""
    pts = pts[-k:]
    f = np.array([p[0] for p in pts], dtype=np.float64)
    a = np.array([p[1] for p in pts], dtype=np.float64)
    c = np.array([p[2] for p in pts], dtype=np.float64)
    if len(pts) < 2 or f[-1] == f[0]:
        return 0.0, 0.0
    return np.polyfit(f, a, 1)[0], np.polyfit(f, c, 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--end_s", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None,
                    help="Exact fps, overriding everything below.")
    ap.add_argument("--duration_inflation_ratio", type=float, default=1.132,
                    help="Same correction as the counting scripts: fps = file fps * ratio. "
                         "Pass 1.0 to use the file's own fps.")
    ap.add_argument("--m_per_px", type=float, default=0.93e-6)
    ap.add_argument("--flow_dir", choices=list(FLOW_VEC), default="down")

    ap.add_argument("--mog2_history", type=int, default=500)
    ap.add_argument("--mog2_varThreshold", type=float, default=8)
    ap.add_argument("--mask_thresh", type=int, default=60)
    ap.add_argument("--learning_rate", type=float, default=0.0005)
    ap.add_argument("--gauss_ksize", type=int, default=3)
    ap.add_argument("--close_iter", type=int, default=1)
    ap.add_argument("--min_area", type=float, default=12)
    ap.add_argument("--max_area", type=float, default=8000)
    ap.add_argument("--use_watershed_split", action="store_true")

    ap.add_argument("--min_step_px", type=float, default=1.0,
                    help="A one-point track only continues forward by at least this much "
                         "per frame, so stationary objects can't start a moving track. "
                         "Cells slower than this (~19 um/s at 0.93 um/px, 20.8 fps) are "
                         "not tracked.")
    ap.add_argument("--max_step_px", type=float, default=70.0,
                    help="Largest along-flow step per frame considered at all.")
    ap.add_argument("--new_lane_px", type=float, default=6.0,
                    help="One-point tracks: max across-flow drift per frame.")
    ap.add_argument("--along_tol_px", type=float, default=4.0,
                    help="Established tracks: along-flow prediction tolerance floor (px).")
    ap.add_argument("--along_tol_frac", type=float, default=0.3,
                    help="Established tracks: along-flow tolerance also scales with speed "
                         "(this fraction of the predicted step).")
    ap.add_argument("--across_tol_px", type=float, default=4.0,
                    help="Established tracks: across-flow prediction tolerance (px).")
    ap.add_argument("--max_missed", type=int, default=2,
                    help="Frames a track may coast along its prediction without a match.")
    ap.add_argument("--vel_window", type=int, default=5,
                    help="Points used for each track's velocity estimate.")
    ap.add_argument("--min_confirm", type=int, default=4,
                    help="Points needed before a track is reported (the 3rd+ points must "
                         "have matched the track's own prediction).")
    ap.add_argument("--prior_frames", type=int, default=150,
                    help="Frames used for the tracking-free speed prior / reference.")

    ap.add_argument("--max_reliable_um_s", type=float, default=800.0,
                    help="Tracks faster than this get speed_reliable=0. Default from the "
                         "stride test on 5_minute.mp4 (compare_stride_runs.py): step "
                         "agreement 77-98%% up to 800 um/s, ~50%% above -- above this, "
                         "cells move farther per frame than the spacing between cells.")
    ap.add_argument("--min_moving_um_s", type=float, default=20.0,
                    help="Tracks slower than this get is_stationary=1: cells sitting on the "
                         "surface or debris, tracked from detection jitter. Not free-flowing "
                         "cells -- exclude them from velocity/height analysis (on 5_minute.mp4, "
                         "40-60 s: ~9%% of tracks). Shown in gray in --out_video.")
    ap.add_argument("--core_size_frac", type=float, default=0.35,
                    help="Cell size from each detection's bright core (same method and "
                         "default as the counting scripts' mean_r).")
    ap.add_argument("--contact_radius_factor", type=float, default=1.8,
                    help="in_contact=1 when est_height_naive_um <= this x mean_r_um. 1.8 is "
                         "Oh et al. 2015 (J Cell Sci 128:3731), who use the same naive height "
                         "equation (their Eqn 3, wall effects neglected) and count a cell as "
                         "in contact with the substrate when its calculated height is within "
                         "1.8x the cell radius. Because that equation ignores wall drag, a cell "
                         "touching the substrate typically gets a height BELOW its own radius "
                         "-- that is the signature of contact, not an error.")
    ap.add_argument("--shear_stress_pa", type=float, default=None)
    ap.add_argument("--chamber_height_um", type=float, default=None)
    ap.add_argument("--medium_viscosity_pa_s", type=float, default=None,
                    help="Set all three to add height estimates per track (same two models "
                         "as the counting scripts).")
    ap.add_argument("--video_scale", type=int, default=2,
                    help="--out_video is upscaled by this factor so labels are readable.")
    ap.add_argument("--video_fps", type=float, default=None,
                    help="Playback fps of --out_video (default: half the source fps, "
                         "i.e. slow motion, so individual cells can be followed by eye).")
    ap.add_argument("--validate_window", type=int, default=150,
                    help="Frames per validation window. Flow speed changes during a "
                         "recording (measured on 5_minute.mp4: from 0 to ~46 px/frame "
                         "between windows), so tracker speed is compared with the "
                         "tracking-free reference window by window, not as one global "
                         "number. 0 disables validation.")
    ap.add_argument("--out_windows_csv", default="predictive_tracks_windows.csv")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Process only every Nth frame (frame numbers stay true video "
                         "frames, so speeds stay in real units). Used as an aliasing test: "
                         "a real cell's speed is the same at stride 1 and 2, but a track "
                         "that hops to the next cell in a lane each processed frame gains "
                         "(cell spacing / stride) per frame, so its speed changes with "
                         "stride. Compare runs with compare_stride_runs.py.")
    ap.add_argument("--out_points_csv", default=None,
                    help="Optional: every tracked point (track_id, frame, x, y) of every "
                         "reported track, needed to match tracks between runs.")
    ap.add_argument("--out_csv", default="predictive_tracks.csv")
    ap.add_argument("--out_video", default=None,
                    help="Optional annotated video with each track's trail.")
    args = ap.parse_args()

    det_args = types.SimpleNamespace(**DETECT_DEFAULTS)
    det_args.min_area, det_args.max_area = args.min_area, args.max_area
    det_args.use_watershed_split = args.use_watershed_split

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fps = args.fps or file_fps * args.duration_inflation_ratio
    um_per_px = args.m_per_px * 1e6
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(max(0.0, args.start_s) * fps)
    end_frame = int(args.end_s * fps) if args.end_s else None

    shifts, spectrum = xcorr_speed_spectrum(args.video, start_frame, args.prior_frames,
                                            args.flow_dir, int(args.max_step_px))
    xcorr_peak_px = float(shifts[np.argmax(spectrum)])
    print(f"fps={fps:.3f}  initial speed prior (tracking-free, first {args.prior_frames} frames): "
          f"{xcorr_peak_px:.0f} px/frame; afterwards the prior follows confirmed tracks")

    recent_v = deque(maxlen=200)
    v_prior = max(xcorr_peak_px, args.min_step_px)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=args.mog2_history, varThreshold=args.mog2_varThreshold, detectShadows=False)
    vw = None
    S = max(1, args.video_scale)
    if args.out_video:
        vw = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.video_fps or file_fps / 2.0, (W * S, H * S))

    tracks, finished = {}, []
    next_id, frame_idx = 1, start_frame
    # along-flow coordinate of the exit edge, and the frame's extent along the flow
    fa_limit = {"down": H, "right": W, "up": 0, "left": 0}[args.flow_dir]
    flow_extent = H if args.flow_dir in ("down", "up") else W

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start_frame) % args.frame_stride:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_sharp = gray  # unblurred, for cell-size measurement
        if args.gauss_ksize > 0:
            k = args.gauss_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        # keep the background model's time constant in real seconds when skipping frames
        fg = backsub.apply(gray, learningRate=min(1.0, args.learning_rate * args.frame_stride))
        _, th = cv2.threshold(fg, args.mask_thresh, 255, cv2.THRESH_BINARY)
        if args.close_iter > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=args.close_iter)
        _, detections, det_boxes = detect_cells(th, frame, gray, det_args)

        def point(j):
            bx, by, bw, bh = det_boxes[j][:4]
            cw, ch = measure_core_size(gray_sharp, bx, by, bw, bh, args.core_size_frac)
            return (frame_idx, D[j, 0], D[j, 1], detections[j][0], detections[j][1], (cw + ch) / 4.0)

        D = np.array([to_flow(d[0], d[1], args.flow_dir) for d in detections], dtype=np.float64).reshape(-1, 2)
        free = set(range(len(D)))

        # Pass 1: established tracks (>=2 points) against their own prediction.
        est = [t for t, st in tracks.items() if len(st["pts"]) >= 2]
        if est and free:
            cols = sorted(free)
            cost = np.full((len(est), len(cols)), np.inf)
            for r, t in enumerate(est):
                st = tracks[t]
                gap = frame_idx - st["pts"][-1][0]
                va, vc = st["v"]
                pa = st["pts"][-1][1] + va * gap
                pc = st["pts"][-1][2] + vc * gap
                sa = max(args.along_tol_px, args.along_tol_frac * abs(va) * gap)
                sc = args.across_tol_px
                ea = (D[cols, 0] - pa) / sa
                ec = (D[cols, 1] - pc) / sc
                cost[r] = ea ** 2 + ec ** 2
            for r, c in assign(cost, gate=1.0):
                j = cols[c]
                st = tracks[est[r]]
                st["pts"].append(point(j))
                st["v"] = track_velocity([(p[0], p[1], p[2]) for p in st["pts"]], args.vel_window)
                st["missed"] = 0
                free.discard(j)

        # Pass 2: one-point tracks -- forward only, narrow lane, prefer the prior speed.
        new = [t for t, st in tracks.items() if len(st["pts"]) == 1 and st["pts"][-1][0] < frame_idx]
        if new and free:
            cols = sorted(free)
            cost = np.full((len(new), len(cols)), np.inf)
            for r, t in enumerate(new):
                f0, a0, c0 = tracks[t]["pts"][-1][:3]
                gap = frame_idx - f0
                da = D[cols, 0] - a0
                dc = D[cols, 1] - c0
                ok_ = ((da >= args.min_step_px * gap) & (da <= args.max_step_px * gap)
                       & (np.abs(dc) <= args.new_lane_px * gap))
                c = ((da - v_prior * gap) / (0.5 * v_prior * gap + 1e-9)) ** 2 + (dc / (args.new_lane_px * gap)) ** 2
                cost[r] = np.where(ok_, c, np.inf)
            for r, c in assign(cost, gate=1e6):
                if not np.isfinite(cost[r, c]):
                    continue
                j = cols[c]
                st = tracks[new[r]]
                st["pts"].append(point(j))
                st["v"] = track_velocity([(p[0], p[1], p[2]) for p in st["pts"]], args.vel_window)
                st["missed"] = 0
                free.discard(j)

        for j in free:
            tracks[next_id] = dict(pts=[point(j)],
                                   v=(0.0, 0.0), missed=0)
            next_id += 1

        for t in list(tracks):
            st = tracks[t]
            if st["pts"][-1][0] == frame_idx:
                continue
            st["missed"] += 1
            gap = frame_idx - st["pts"][-1][0]
            predicted_along = st["pts"][-1][1] + st["v"][0] * gap
            leaving = predicted_along > fa_limit
            limit = 0 if len(st["pts"]) == 1 else args.max_missed
            if st["missed"] > limit or leaving:
                done = tracks.pop(t)
                if len(done["pts"]) >= args.min_confirm:
                    finished.append((t, done))
                    recent_v.append(done["v"][0])
                    v_prior = max(float(np.median(recent_v)), args.min_step_px)

        if vw is not None:
            # Only tracks matched on THIS frame are drawn, so every circle sits on a
            # detection the tracker actually used -- nothing is drawn from prediction.
            vis = cv2.resize(frame, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
            for t, st in tracks.items():
                if len(st["pts"]) < 3 or st["pts"][-1][0] != frame_idx:
                    continue
                hue = int((t * 0.618033988749895) % 1.0 * 179)
                color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
                xy = [(int(p[3] * S), int(p[4] * S)) for p in st["pts"][-40:]]
                for p0, p1 in zip(xy, xy[1:]):
                    cv2.line(vis, p0, p1, color, 1, cv2.LINE_AA)
                for q in xy[:-1]:
                    cv2.circle(vis, q, 1, color, -1)
                speed = st["v"][0] * um_per_px * fps
                reliable = speed <= args.max_reliable_um_s
                if speed < args.min_moving_um_s:
                    color = (150, 150, 150)
                cv2.circle(vis, xy[-1], int(max(4, st["pts"][-1][5]) * S) + 2,
                           color if reliable else (0, 0, 255), 1 if reliable else 2, cv2.LINE_AA)
                cv2.putText(vis, f"{t}:{speed:.0f}" + ("" if reliable else "?"),
                            (xy[-1][0] + 8 * S // 2 + 4, xy[-1][1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.33 * S, color if reliable else (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(vis, (0, 0), (W * S, 18 * S // 2 + 6), (0, 0, 0), -1)
            cv2.putText(vis, f"t={frame_idx / fps:6.2f}s  label = track:speed(um/s)   "
                             f"red ? = >{args.max_reliable_um_s:.0f} um/s (unreliable)   gray = stationary",
                        (6, 9 * S // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * S, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(vis)

        frame_idx += 1

    cap.release()
    if vw is not None:
        vw.release()
        print(f"Wrote: {args.out_video}")
    for t, st in tracks.items():
        if len(st["pts"]) >= args.min_confirm:
            finished.append((t, st))

    height_params = (args.shear_stress_pa, args.chamber_height_um, args.medium_viscosity_pa_s)
    want_height = all(x is not None for x in height_params)
    rows, steps, step_frames = [], [], []
    for t, st in finished:
        f = np.array([p[0] for p in st["pts"]], dtype=np.float64)
        a = np.array([p[1] for p in st["pts"]])
        c = np.array([p[2] for p in st["pts"]])
        slope = np.polyfit(f, a, 1)[0]
        step = np.diff(a) / np.diff(f)
        steps.extend(step.tolist())
        step_frames.extend(f[1:].tolist())
        resid = a - np.polyval(np.polyfit(f, a, 1), f)
        rows.append(dict(
            track_id=t, n_points=len(f), first_frame=int(f[0]), last_frame=int(f[-1]),
            duration_s=(f[-1] - f[0]) / fps,
            start_x=st["pts"][0][3], start_y=st["pts"][0][4],
            end_x=st["pts"][-1][3], end_y=st["pts"][-1][4],
            along_flow_um=(a[-1] - a[0]) * um_per_px,
            across_flow_um=(c[-1] - c[0]) * um_per_px,
            speed_um_s=slope * um_per_px * fps,
            speed_step_cv=float(np.std(step) / np.mean(step)) if np.mean(step) > 0 else float("nan"),
            fit_resid_px=float(np.sqrt(np.mean(resid ** 2))),
            backward_steps=int((step < 0).sum()),
            mean_r_um=float(np.mean([p[5] for p in st["pts"]])) * um_per_px,
            speed_reliable=int(slope * um_per_px * fps <= args.max_reliable_um_s),
            is_stationary=int(slope * um_per_px * fps < args.min_moving_um_s),
        ))
        if want_height:
            v = slope * um_per_px * fps * 1e-6          # m/s
            r_m = rows[-1]["mean_r_um"] * 1e-6
            h_m = args.chamber_height_um * 1e-6
            disc = h_m ** 2 / 4.0 - v * h_m * args.medium_viscosity_pa_s / args.shear_stress_pa
            rows[-1]["est_height_naive_um"] = (h_m / 2.0 - math.sqrt(disc)) * 1e6 if disc >= 0 and v > 0 else float("nan")
            y = solve_height_wall_corrected(v, args.shear_stress_pa / args.medium_viscosity_pa_s, r_m, h_m) if v > 0 else float("nan")
            rows[-1]["est_height_wall_corrected_um"] = y * 1e6 if np.isfinite(y) else float("nan")
            hn = rows[-1]["est_height_naive_um"]
            rows[-1]["in_contact"] = int(np.isfinite(hn) and hn <= args.contact_radius_factor * rows[-1]["mean_r_um"])
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    if args.out_points_csv:
        pd.DataFrame([dict(track_id=t, frame=p[0], x=p[3], y=p[4])
                      for t, st in finished for p in st["pts"]]).to_csv(args.out_points_csv, index=False)
    print(f"Wrote: {args.out_csv}  ({len(df)} confirmed tracks)")
    if not len(df):
        return

    steps, step_frames = np.array(steps), np.array(step_frames)
    to_ums = um_per_px * fps
    print(f"\n  per-track speed, median : {df.speed_um_s.median():.0f} um/s "
          f"(p25 {df.speed_um_s.quantile(.25):.0f}, p75 {df.speed_um_s.quantile(.75):.0f})")
    print(f"  backward steps          : {100 * (steps < 0).mean():.1f}% of all steps")
    print(f"  track length, median    : {df.n_points.median():.0f} points; tracks spanning "
          f">=75% of the frame: {(df.along_flow_um >= 0.75 * flow_extent * um_per_px).sum()}")

    if want_height:
        mv = df[(df.speed_reliable == 1) & (df.is_stationary == 0)]
        if len(mv):
            print(f"  moving, reliably tracked: {len(mv)}; in contact with substrate "
                  f"(naive height <= {args.contact_radius_factor}x radius): {100 * mv.in_contact.mean():.0f}%")

    if args.validate_window <= 0:
        return
    hist_shifts = np.arange(-10, int(args.max_step_px) + 1)
    wrows = []
    for w0 in range(start_frame, frame_idx, args.validate_window):
        w1 = min(w0 + args.validate_window, frame_idx)
        if w1 - w0 < 10:
            break
        s, sc = xcorr_speed_spectrum(args.video, w0, w1 - w0, args.flow_dir, int(args.max_step_px))
        ref = float(s[np.argmax(sc)])
        ws = steps[(step_frames >= w0) & (step_frames < w1)]
        if len(ws):
            counts = np.array([(np.round(ws) == k).sum() for k in hist_shifts])
            mode = float(hist_shifts[np.argmax(counts)])
        else:
            mode = float("nan")
        wrows.append(dict(
            start_s=w0 / fps, end_s=w1 / fps, reference_px_per_frame=ref,
            reference_um_s=ref * to_ums, reference_peak_sharpness=float(sc.max() - np.median(sc)),
            tracked_steps=len(ws),
            tracked_mode_px_per_frame=mode,
            tracked_median_px_per_frame=float(np.median(ws)) if len(ws) else float("nan"),
            tracked_median_um_s=float(np.median(ws)) * to_ums if len(ws) else float("nan"),
        ))
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(args.out_windows_csv, index=False)
    print(f"\nValidation, window by window (tracking-free reference vs tracked steps) -> {args.out_windows_csv}")
    print(wdf[["start_s", "reference_um_s", "tracked_median_um_s", "tracked_steps"]].round(0).to_string(index=False))
    moving = wdf[(wdf.reference_px_per_frame >= 2) & (wdf.tracked_steps >= 50)]
    if len(moving) >= 2:
        ratio = moving.tracked_median_um_s / moving.reference_um_s
        r = np.corrcoef(moving.reference_um_s, moving.tracked_median_um_s)[0, 1]
        print(f"  windows with flow: {len(moving)}; tracked/reference median ratio "
              f"{ratio.median():.2f} (p25 {ratio.quantile(.25):.2f}, p75 {ratio.quantile(.75):.2f}); "
              f"correlation r={r:.2f}")


if __name__ == "__main__":
    main()
