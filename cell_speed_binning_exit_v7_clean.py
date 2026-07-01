"""
v7_clean — back to v7 base that worked best, with two additions:
  --erode_iter N : erode the threshold mask before contour detection.
  This separates touching bright halos without watershed complexity.
  Start with --erode_iter 1 or 2.

  --use_watershed_split : split touching/overlapping cells via a
  distance-transform watershed (requires scikit-image). See
  split_contours_watershed() for why this replaced the earlier
  cv2.watershed-based approach, which produced unreliable splits.

  All other v7 logic preserved exactly.
"""

import argparse, math
from collections import deque, defaultdict
import cv2
import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as sk_watershed
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def parse_bins(s):
    edges = [float(x) for x in s.split(",") if x.strip() != ""]
    if len(edges) < 2:
        raise ValueError("speed_bins needs at least 2 edges")
    return edges  # plain Python list, not numpy array


def bin_index(v, edges):
    v = float(v)  # ensure plain Python float, not numpy scalar
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    if v >= edges[-1]:
        return len(edges) - 1
    return None


def hungarian_match(tracks, detections, max_dist):
    tids = list(tracks.keys())
    n_t, n_d = len(tids), len(detections)
    if n_t == 0 or n_d == 0:
        return [], set(range(n_d))

    if not HAS_SCIPY:
        used, pairs = set(), []
        for tid in tids:
            st = tracks[tid]
            best_j, best_d = None, 1e9
            for j, (cx, cy, _) in enumerate(detections):
                if j in used:
                    continue
                d = math.hypot(cx - st["cx"], cy - st["cy"])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= max_dist:
                pairs.append((tid, best_j))
                used.add(best_j)
        return pairs, set(range(n_d)) - {j for _, j in pairs}

    INF = 1e9
    cost = np.full((n_t, n_d), INF, dtype=np.float64)
    for i, tid in enumerate(tids):
        st = tracks[tid]
        for j, (cx, cy, _) in enumerate(detections):
            d = math.hypot(cx - st["cx"], cy - st["cy"])
            if d <= max_dist:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs, matched = [], set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF:
            pairs.append((tids[i], j))
            matched.add(j)
    return pairs, set(range(n_d)) - matched


def split_contours_watershed(binary_mask, frame_bgr, fg_thresh, min_area, min_split_area,
                              min_peak_dist):
    """
    Split touching/overlapping cells using a distance-transform watershed.

    Seeds are the distance-transform local maxima (cell centers), found with
    skimage.feature.peak_local_max so that two seeds separated by at least
    min_peak_dist are always kept distinct (a plain dilation-based local-max
    test merges seeds that sit close together, which is exactly the touching-
    cell case this is meant to fix). Flooding uses skimage.segmentation.watershed
    on -dist restricted to the blob mask, so basins sit at the cell centers and
    the split line falls on the true ridge between them. cv2.watershed is not
    used here: its flooding order is driven by local gradient magnitude, which
    is nearly flat across a smooth distance-transform surface, so it produces
    essentially arbitrary (often massively lopsided) splits on this input.
    """
    mask = binary_mask.copy()
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for rc in raw_contours:
        area = cv2.contourArea(rc)
        if area < min_area:
            continue

        # small blobs → single cell, skip splitting
        if area < min_split_area:
            out.append(rc)
            continue

        # isolate blob
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [rc], -1, 255, -1)

        dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            out.append(rc)
            continue

        coords = peak_local_max(dist, min_distance=max(1, int(min_peak_dist)),
                                 threshold_abs=fg_thresh * dist.max(),
                                 labels=blob_mask)

        if len(coords) <= 1:
            # only one cell center found → single cell
            out.append(rc)
            continue

        peak_mask = np.zeros(dist.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, n_seeds = ndi.label(peak_mask)

        labels = sk_watershed(-dist, markers, mask=blob_mask)

        split_any = False
        for label in range(1, n_seeds + 1):
            obj = np.uint8(labels == label) * 255
            cs, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) >= min_area:
                    out.append(c)
                    split_any = True

        if not split_any:
            out.append(rc)

    return out


def detect_cells(th, frame, gray, args, fg_thresh=None, min_peak_dist=None):
    """
    Mask -> contours -> filtered (cx, cy, is_streak) detections. Shared by the
    real per-frame loop and the quick preview/sweep path so both always see
    identical detection behavior. fg_thresh / min_peak_dist let the preview
    path override the CLI values per-tile without touching args.
    """
    if args.use_watershed_split:
        min_split = args.watershed_min_split_area or (1.8 * args.min_area)
        if fg_thresh is None:
            fg_thresh = args.watershed_fg_thresh
        if min_peak_dist is None:
            min_peak_dist = args.watershed_min_peak_dist or math.sqrt(min_split / (2 * math.pi))
        contours = split_contours_watershed(th, frame, fg_thresh, args.min_area,
                                            min_split_area=min_split,
                                            min_peak_dist=min_peak_dist)
    else:
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    det_boxes  = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < args.min_area or area > args.max_area:
            continue
        if args.min_mean_intensity > 0:
            blob_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=blob_mask)[0]
            if mean_intensity < args.min_mean_intensity:
                continue  # dark/black blob (e.g. debris, dead cell) - not counted
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(c)
        long_axis  = float(max(w, h))
        short_axis = float(max(1, min(w, h)))
        is_streak  = (args.enable_streak
                      and area >= args.streak_min_area
                      and long_axis >= args.streak_min_len
                      and long_axis/short_axis >= args.streak_ar)
        detections.append((cx, cy, is_streak))
        det_boxes.append((x, y, w, h, is_streak))
    return contours, detections, det_boxes


def print_blob_size_diagnostics(th, args):
    """
    Report the actual raw-blob-area distribution in this frame (before any
    --min_area/--max_area/watershed filtering) and back out suggested values
    for --min_area / --watershed_min_split_area / --watershed_min_peak_dist
    from it, so those don't have to be guessed from a screenshot. A blob-size
    mismatch here (e.g. --min_area tuned for a few-px speck when real cells
    are tens of px across) is what causes watershed to shred single cells
    into dozens of pieces -- watershed_min_peak_dist auto-derives from
    min_area, so an undersized min_area makes it absurdly small.
    """
    raw_contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in raw_contours])
    areas = areas[areas > 0]
    if len(areas) == 0:
        print("No foreground blobs found in this frame (mask is empty) -- check "
              "--mask_thresh / --mog2_varThreshold, or that warmup has enough frames.")
        return
    p10, p25, med, p75, p90 = np.percentile(areas, [10, 25, 50, 75, 90])
    print(f"\nRaw blob areas this frame, before --min_area/--max_area filtering "
          f"(n={len(areas)} blobs):")
    print(f"  min={areas.min():.0f}  p10={p10:.0f}  p25={p25:.0f}  median={med:.0f}  "
          f"p75={p75:.0f}  p90={p90:.0f}  max={areas.max():.0f}")
    suggested_min_area  = max(1, round(med * 0.3))
    suggested_min_split = round(med * 1.5)
    suggested_peak_dist = max(1, round(math.sqrt(med / (2 * math.pi))))
    print(f"  Current --min_area={args.min_area:g}. If the median blob (~{med:.0f}px²) "
          f"is roughly one cell, try:")
    print(f"    --min_area {suggested_min_area} --watershed_min_split_area "
          f"{suggested_min_split} --watershed_min_peak_dist {suggested_peak_dist}")
    print(f"  (median is only a rough estimate -- it's skewed up if many blobs here are "
          f"already touching pairs, and down if there's a lot of small debris)\n")


def save_preview(th, frame, gray, args):
    """
    Annotate one already-computed frame/mask with detected cell counts and
    save it, instead of writing a full-video debug file. If --sweep_fg_thresh
    and/or --sweep_min_peak_dist give more than one value, builds a grid with
    one tile per combination (rows = min_peak_dist, cols = fg_thresh) so
    several settings can be compared at a glance from a single frame.
    """
    print_blob_size_diagnostics(th, args)

    if not args.use_watershed_split and (args.sweep_fg_thresh or args.sweep_min_peak_dist):
        print("Note: --sweep_fg_thresh/--sweep_min_peak_dist only affect anything "
              "when --use_watershed_split is also passed.")

    fg_list = ([float(x) for x in args.sweep_fg_thresh.split(",")] if args.sweep_fg_thresh
               else [args.watershed_fg_thresh])
    if args.sweep_min_peak_dist:
        pd_list = [float(x) for x in args.sweep_min_peak_dist.split(",")]
    else:
        min_split = args.watershed_min_split_area or (1.8 * args.min_area)
        pd_list = [args.watershed_min_peak_dist or math.sqrt(min_split / (2 * math.pi))]

    print(f"{'fg_thresh':>10} {'min_peak_dist':>14} {'count':>6}")
    rows = []
    for pd in pd_list:
        tiles = []
        for fg in fg_list:
            _, detections, _ = detect_cells(th, frame, gray, args, fg_thresh=fg, min_peak_dist=pd)
            print(f"{fg:>10.2f} {pd:>14.2f} {len(detections):>6}")
            tile = frame.copy()
            for i, (cx, cy, _is_streak) in enumerate(detections):
                cv2.circle(tile, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(tile, str(i + 1), (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(tile, f"fg={fg:g} d={pd:g} n={len(detections)}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    cv2.imwrite(args.preview_out, grid)
    print(f"Wrote: {args.preview_out}  ({len(pd_list)}x{len(fg_list)} grid)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start_s",  type=float, default=0.0)
    ap.add_argument("--end_s",    type=float, default=None)
    ap.add_argument("--warmup_s", type=float, default=0.0)
    ap.add_argument("--bin_seconds", type=float, default=1.0)
    ap.add_argument("--speed_bins", required=True)
    ap.add_argument("--m_per_px", type=float, default=None)
    ap.add_argument("--fps",      type=float, default=None)

    ap.add_argument("--min_area",   type=float, default=12)
    ap.add_argument("--max_area",   type=float, default=8000)
    ap.add_argument("--max_dist",   type=float, default=220)
    ap.add_argument("--max_missed", type=int,   default=20)
    ap.add_argument("--min_mean_intensity", type=float, default=0.0,
                    help="Minimum mean grayscale intensity (0-255) inside a detected blob "
                         "for it to be counted as a cell. Blobs darker than this ('black' "
                         "cells, debris, dead cells) are discarded before tracking. "
                         "Default 0 = no filtering. Use --preview_frame_s to check a value "
                         "against a real frame before committing to a full run.")

    ap.add_argument("--mog2_history",      type=int,   default=500)
    ap.add_argument("--mog2_varThreshold", type=float, default=16)
    ap.add_argument("--learning_rate",     type=float, default=0.0005)
    ap.add_argument("--mask_thresh",       type=int,   default=60)
    ap.add_argument("--gauss_ksize",       type=int,   default=3)

    ap.add_argument("--use_framediff",   action="store_true")
    ap.add_argument("--diff_thresh",     type=int, default=12)
    ap.add_argument("--diff_close_iter", type=int, default=1)
    ap.add_argument("--framediff_mode", type=str, default="and_fallback",
                    choices=["or", "and", "and_fallback"],
                    help="How to combine MOG2 and framediff masks. "
                         "'or' = original (catches more, merges more). "
                         "'and' = intersection (clean separation, may miss some). "
                         "'and_fallback' = AND unless too few blobs, then OR (default).")
    ap.add_argument("--open_iter",       type=int, default=0)
    ap.add_argument("--close_iter",      type=int, default=1)

    # KEY NEW PARAM: erode before contour to separate touching halos
    ap.add_argument("--erode_iter", type=int, default=0,
                    help="Erode threshold mask before contour detection. "
                         "1-2 iterations separates touching bright halos. "
                         "Does not affect exit line logic.")
    ap.add_argument("--bridge_break_size", type=int, default=0,
                    help="Tiny erosion (kernel size in px, e.g. 3) applied ONLY to break "
                         "1-2px bridges between adjacent blobs that findContours merges "
                         "into one contour, then immediately dilated back to restore "
                         "original blob size. Unlike --erode_iter, this does not shrink "
                         "final blob area. Try 3 first.")

    ap.add_argument("--use_watershed_split",  action="store_true")
    ap.add_argument("--watershed_fg_thresh",  type=float, default=0.7)
    ap.add_argument("--watershed_min_split_area", type=float, default=None,
                    help="Only attempt watershed on blobs larger than this area (px²). "
                         "Blobs smaller than this are treated as single cells and skipped. "
                         "Default: 1.8 * min_area. Set to ~1.5x typical single-cell area.")
    ap.add_argument("--watershed_min_peak_dist", type=float, default=None,
                    help="Minimum distance (px) between two cell-center seeds for them to "
                         "be split into separate cells. Two touching cells closer together "
                         "than this are kept as one. Default: radius of a single cell, "
                         "estimated as sqrt(min_split_area / (2*pi)).")

    ap.add_argument("--preview_frame_s", type=float, default=None,
                    help="Quick-check mode: instead of processing the whole video, grab the "
                         "frame this many seconds after --start_s, run detection on it, print "
                         "the resulting cell count, save an annotated image to --preview_out, "
                         "and exit. Runs in a fraction of a second instead of a full pass.")
    ap.add_argument("--preview_out", default="preview.png",
                    help="Where to save the annotated image for --preview_frame_s.")
    ap.add_argument("--sweep_fg_thresh", type=str, default=None,
                    help="Comma-separated watershed_fg_thresh values to compare side by side "
                         "in the --preview_frame_s image (e.g. '0.5,0.6,0.7,0.8'). "
                         "Default: just --watershed_fg_thresh.")
    ap.add_argument("--sweep_min_peak_dist", type=str, default=None,
                    help="Comma-separated watershed_min_peak_dist values to compare side by "
                         "side in the --preview_frame_s image (e.g. '3,5,7,10'). "
                         "Default: just --watershed_min_peak_dist / its auto default.")

    ap.add_argument("--min_track_frames_for_speed", type=int, default=3)
    ap.add_argument("--allow_single_frame_count",   action="store_true")

    ap.add_argument("--enable_streak",       action="store_true")
    ap.add_argument("--streak_ar",           type=float, default=2.2)
    ap.add_argument("--streak_min_len",      type=float, default=4)
    ap.add_argument("--streak_min_area",     type=float, default=6)
    ap.add_argument("--streak_merge_dist",   type=float, default=25)
    ap.add_argument("--streak_merge_window", type=int,   default=2)

    ap.add_argument("--dedup_window", type=float, default=0.15)
    ap.add_argument("--dedup_dist",   type=float, default=25)
    ap.add_argument("--dedup_keep",   type=int,   default=4000)

    ap.add_argument("--exit_side",      type=str, default="right",
                    choices=["right","left","top","bottom"])
    ap.add_argument("--exit_margin_px", type=int, default=10)
    ap.add_argument("--end_of_range_margin_px", type=int, default=60,
                    help="When the analyzed range ends, a still-active track is only "
                         "counted if it's within this many px of the exit boundary "
                         "(catches cells that were genuinely about to cross when the clip "
                         "cut off). Tracks further from the boundary than this are dropped, "
                         "not counted -- they clearly hadn't reached the exit within the "
                         "observed window, e.g. a cell sitting in the middle of the frame "
                         "the whole time. Should be noticeably larger than --exit_margin_px "
                         "(which is checked every frame already) or nothing extra gets "
                         "caught; set to 0 to disable end-of-range counting entirely.")

    ap.add_argument("--draw_scalebar", action="store_true")
    ap.add_argument("--scalebar_um",   type=float, default=10.0)
    ap.add_argument("--um_per_px",     type=float, default=0.91)

    ap.add_argument("--no_pad_to_range_end", action="store_true")
    ap.add_argument("--out_csv",        default="speed_counts_per_time.csv")
    ap.add_argument("--per_object_csv", default=None)
    ap.add_argument("--debug_video",    default=None)
    ap.add_argument("--debug_show_mask",action="store_true")
    ap.add_argument("--use_edge_barrier", action="store_true",
                    help="Use Canny edge detection on original grayscale to find cell "
                         "boundaries, then subtract them from the mask before contour "
                         "detection. Separates touching cells that share a visible dark "
                         "boundary ring in the raw image.")
    ap.add_argument("--edge_canny_low",  type=int, default=20,
                    help="Canny lower threshold. Default 20.")
    ap.add_argument("--edge_canny_high", type=int, default=60,
                    help="Canny upper threshold. Default 60.")
    ap.add_argument("--edge_dilate",     type=int, default=1,
                    help="Dilate edges by this many px before subtracting from mask. "
                         "1 = thin barrier, 2 = thicker. Default 1.")
    ap.add_argument("--draw_detections",action="store_true")
    args = ap.parse_args()

    if args.use_watershed_split and not HAS_SKIMAGE:
        raise SystemExit(
            "--use_watershed_split requires scikit-image. Install it with: "
            "pip install scikit-image")

    edges = parse_bins(args.speed_bins)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open: {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_s     = max(0.0, args.start_s)
    start_frame = int(start_s * fps)
    end_frame_excl = int(args.end_s * fps) if args.end_s else None
    warmup_frames  = int(max(0.0, args.warmup_s) * fps)
    warmup_start   = max(0, start_frame - warmup_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    ret, frame0 = cap.read()
    if not ret:
        raise SystemExit("Cannot read first frame.")
    H, W = frame0.shape[:2]
    scalebar_px = max(1, int(round(args.scalebar_um / args.um_per_px)))

    if args.preview_frame_s is not None:
        total_frames  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_len_s    = (total_frames / fps) if total_frames > 0 else None
        target_abs_s   = start_s + args.preview_frame_s
        frames_to_scan = int(args.preview_frame_s * fps) + warmup_frames
        if video_len_s is not None and target_abs_s > video_len_s:
            print(f"WARNING: --preview_frame_s {args.preview_frame_s:g} (+ --start_s "
                  f"{start_s:g}) = {target_abs_s:.1f}s, but the video is only "
                  f"~{video_len_s:.1f}s long. --preview_frame_s counts SECONDS after "
                  f"--start_s, not a frame number. The preview target will never be "
                  f"reached, so this will scan the ENTIRE video (same cost as a full run) "
                  f"before falling through to normal output instead of a quick preview.")
        print(f"Quick check: scanning ~{frames_to_scan} frame(s) "
              f"(~{frames_to_scan/fps:.1f}s of video) before the preview point...")

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=args.mog2_history,
        varThreshold=args.mog2_varThreshold,
        detectShadows=False)

    next_id  = 1
    tracks   = {}
    recent_streaks = deque()
    recent_exits   = deque()
    counts             = defaultdict(int)
    streak_only_counts = defaultdict(int)
    short_track_counts = defaultdict(int)
    per_rows = []

    vw = None
    if args.debug_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w  = (2 * W) if args.debug_show_mask else W
        vw     = cv2.VideoWriter(args.debug_video, fourcc, fps, (out_w, H))

    prev_gray = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    abs_frame = warmup_start
    last_processed = None

    def crossed_exit(cx, cy):
        m = args.exit_margin_px
        if args.exit_side == "right":  return cx >= W - 1 - m
        if args.exit_side == "left":   return cx <= m
        if args.exit_side == "bottom": return cy >= H - 1 - m
        return cy <= m

    def near_exit(cx, cy, m):
        if args.exit_side == "right":  return cx >= W - 1 - m
        if args.exit_side == "left":   return cx <= m
        if args.exit_side == "bottom": return cy >= H - 1 - m
        return cy <= m

    def is_duplicate_exit(t, x, y):
        # Drop stale entries by scanning the whole buffer rather than just the front:
        # a lingering cell refreshes its entry's timestamp below, so the buffer is not
        # guaranteed sorted by time and a front-only pop would leave old entries stuck.
        kept = [(t0, x0, y0) for (t0, x0, y0) in recent_exits if t - t0 <= args.dedup_window]
        for i, (t0, x0, y0) in enumerate(kept):
            if math.hypot(x - x0, y - y0) <= args.dedup_dist:
                # Refresh instead of leaving the original timestamp: a cell that just
                # sits in the exit margin re-triggers crossed_exit() every frame (its
                # track gets deleted and immediately recreated), and without refreshing,
                # this entry would expire after exactly one dedup_window and let the
                # same still-lingering cell be counted again as a new exit.
                kept[i] = (t, x0, y0)
                recent_exits.clear()
                recent_exits.extend(kept)
                return True
        recent_exits.clear()
        recent_exits.extend(kept)
        return False

    def compute_speed(st):
        dt = max(1, st["last_seen_frame"] - st["first_frame"]) / fps
        v  = math.hypot(float(st["end_x"]) - float(st["start_x"]),
                        float(st["end_y"]) - float(st["start_y"])) / dt
        return v * args.m_per_px if args.m_per_px else v

    def finalize_track(t_abs, t_rel, st, tid, reason):
        x_e, y_e = float(st["end_x"]), float(st["end_y"])
        if is_duplicate_exit(t_abs, x_e, y_e):
            return False
        recent_exits.append((t_abs, x_e, y_e))
        if len(recent_exits) > args.dedup_keep:
            recent_exits.popleft()
        tb  = int(t_rel // args.bin_seconds)
        thr = 1 if (args.allow_single_frame_count and reason == "passed_line")               else args.min_track_frames_for_speed
        if st["seen_count"] >= thr:
            v  = float(compute_speed(st))
            sb = bin_index(v, edges)
            if sb is not None:
                counts[(tb, sb)] += 1
            if args.per_object_csv:
                per_rows.append(dict(
                    track_id=tid, final_reason=reason,
                    seen_count=st["seen_count"], is_streak=int(st["is_streak"]),
                    speed=v, speed_unit="m/s" if args.m_per_px else "px/s",
                    time_exit_s_abs=t_abs, time_exit_s_rel=t_rel,
                    x_exit=x_e, y_exit=y_e))
        else:
            bucket = streak_only_counts if st["is_streak"] else short_track_counts
            bucket[tb] += 1
        return True

    while True:
        if end_frame_excl is not None and abs_frame >= end_frame_excl:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cur_frame = abs_frame
        abs_frame += 1
        last_processed = cur_frame

        if args.preview_frame_s is not None and cur_frame % 200 == 0:
            print(f"  ...scanned frame {cur_frame} (t={cur_frame/fps:.1f}s), "
                  f"target {start_s + args.preview_frame_s:.1f}s", flush=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.gauss_ksize > 0:
            k = args.gauss_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        fg = backsub.apply(gray, learningRate=float(args.learning_rate))
        _, th_mog2 = cv2.threshold(fg, args.mask_thresh, 255, cv2.THRESH_BINARY)

        th_diff = None
        if args.use_framediff and prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, th_diff = cv2.threshold(diff, args.diff_thresh, 255, cv2.THRESH_BINARY)
            if args.diff_close_iter > 0:
                th_diff = cv2.morphologyEx(th_diff, cv2.MORPH_CLOSE,
                                           np.ones((3,3),np.uint8),
                                           iterations=args.diff_close_iter)
        prev_gray = gray

        if cur_frame < start_frame:
            continue

        if th_diff is None:
            th = th_mog2
        elif args.framediff_mode == "and":
            # AND: only pixels detected by BOTH MOG2 and framediff
            # → MOG2 gives clean separation, framediff recovers missed cells
            th = cv2.bitwise_and(th_mog2, th_diff)
        elif args.framediff_mode == "or":
            # OR: original behavior
            th = cv2.bitwise_or(th_mog2, th_diff)
        else:
            # and_fallback: AND first; if result has too few blobs, fall back to OR
            th_and = cv2.bitwise_and(th_mog2, th_diff)
            th_or  = cv2.bitwise_or(th_mog2, th_diff)
            n_and, _ = cv2.findContours(th_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            n_or,  _ = cv2.findContours(th_or,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            th = th_and if len(n_and) >= len(n_or) * 0.5 else th_or

        if args.open_iter > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                                  np.ones((3,3),np.uint8), iterations=args.open_iter)
        if args.close_iter > 0:
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                                  np.ones((3,3),np.uint8), iterations=args.close_iter)

        # NEW: break thin 1-2px bridges between adjacent blobs that findContours
        # would otherwise merge into one contour. Erode then dilate back to
        # restore original blob area (unlike --erode_iter which permanently shrinks).
        if args.bridge_break_size > 0:
            k = args.bridge_break_size | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            th_eroded = cv2.erode(th, kernel, iterations=1)
            th = cv2.dilate(th_eroded, kernel, iterations=1)

        # erode to separate touching halos before contour detection
        if args.erode_iter > 0:
            th = cv2.erode(th, np.ones((3,3),np.uint8), iterations=args.erode_iter)

        # Edge barrier: subtract cell boundary edges from mask to separate touching cells
        if args.use_edge_barrier:
            edges = cv2.Canny(gray, args.edge_canny_low, args.edge_canny_high)
            if args.edge_dilate > 0:
                edges = cv2.dilate(edges, np.ones((3, 3), np.uint8),
                                   iterations=args.edge_dilate)
            th = cv2.bitwise_and(th, cv2.bitwise_not(edges))

        if args.preview_frame_s is not None and (cur_frame / fps - start_s) >= args.preview_frame_s:
            save_preview(th, frame, gray, args)
            cap.release()
            if vw: vw.release()
            return

        contours, detections, det_boxes = detect_cells(th, frame, gray, args)

        for tid in tracks:
            tracks[tid]["updated"] = False

        pairs, unmatched = hungarian_match(tracks, detections, args.max_dist)

        for tid, j in pairs:
            cx, cy, is_streak = detections[j]
            st = tracks[tid]
            st["cx"], st["cy"] = cx, cy
            st["last_seen_frame"] = cur_frame
            st["seen_count"]  += 1
            st["missed_count"] = 0
            st["updated"]      = True
            st["is_streak"]    = st["is_streak"] or is_streak
            st["end_x"], st["end_y"] = cx, cy

        for j in unmatched:
            cx, cy, is_streak = detections[j]
            if args.enable_streak and is_streak:
                while (recent_streaks
                       and cur_frame - recent_streaks[0][0] > args.streak_merge_window):
                    recent_streaks.popleft()
                if any(math.hypot(cx-x0,cy-y0) <= args.streak_merge_dist
                       for _,x0,y0 in recent_streaks):
                    continue
                recent_streaks.append((cur_frame, cx, cy))
            tracks[next_id] = dict(
                cx=cx, cy=cy, first_frame=cur_frame, last_seen_frame=cur_frame,
                seen_count=1, missed_count=0, is_streak=bool(is_streak),
                updated=True, start_x=cx, start_y=cy, end_x=cx, end_y=cy)
            next_id += 1

        to_del = []
        for tid, st in tracks.items():
            if crossed_exit(st["cx"], st["cy"]):
                t_abs = st["last_seen_frame"] / fps
                finalize_track(t_abs, t_abs-start_s, st, tid, "passed_line")
                to_del.append(tid)
        for tid in to_del:
            tracks.pop(tid, None)

        for tid, st in list(tracks.items()):
            if not st["updated"]:
                st["missed_count"] += 1
                if st["missed_count"] > args.max_missed:
                    t_abs = st["last_seen_frame"] / fps
                    finalize_track(t_abs, t_abs-start_s, st, tid, "missing")
                    tracks.pop(tid, None)

        if vw is not None:
            left = frame.copy()
            m  = args.exit_margin_px
            lc = (0, 200, 255)
            if   args.exit_side=="right":  cv2.line(left,(W-1-m,0),(W-1-m,H-1),lc,1)
            elif args.exit_side=="left":   cv2.line(left,(m,0),(m,H-1),lc,1)
            elif args.exit_side=="bottom": cv2.line(left,(0,H-1-m),(W-1,H-1-m),lc,1)
            else:                          cv2.line(left,(0,m),(W-1,m),lc,1)

            if args.draw_scalebar:
                x2,yb = W-20,H-20; x1=max(10,x2-scalebar_px)
                cv2.line(left,(x1,yb),(x2,yb),(255,255,255),2)
                cv2.putText(left,f"{args.scalebar_um:g}um",(x1,yb-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

            if args.draw_detections:
                for (x,y,w,h,is_s) in det_boxes:
                    col = (0,255,255) if is_s else (255,0,255)
                    cv2.rectangle(left,(x,y),(x+w,y+h),col,1)

            for tid,st in tracks.items():
                col = (0,255,0) if not st["is_streak"] else (0,255,255)
                cv2.circle(left,(st["cx"],st["cy"]),3,col,-1)
                cv2.putText(left,str(tid),(st["cx"]+4,st["cy"]-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

            cv2.putText(left,
                f"t={cur_frame/fps:.1f}s det={len(detections)} trk={len(tracks)}",
                (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            if args.debug_show_mask:
                vw.write(np.hstack([left, cv2.cvtColor(th,cv2.COLOR_GRAY2BGR)]))
            else:
                vw.write(left)

    cap.release()
    if vw: vw.release()

    # A track still active when the analyzed range ends (never crossed the exit line,
    # never missed enough frames) is only counted if it's actually near the exit
    # boundary -- i.e. it was genuinely about to cross when the clip cut off. A track
    # anywhere else in the frame (e.g. a cell just sitting/drifting mid-frame the whole
    # window) is dropped, not counted: it never demonstrated it was exiting at all.
    if last_processed is not None and args.end_of_range_margin_px > 0:
        t_abs_end = last_processed / fps
        for tid, st in list(tracks.items()):
            if near_exit(st["cx"], st["cy"], args.end_of_range_margin_px):
                finalize_track(t_abs_end, t_abs_end - start_s, st, tid, "end_of_range")
        tracks.clear()

    if last_processed is None:
        duration_rel_s = 0.0
    elif end_frame_excl:
        duration_rel_s = (end_frame_excl - start_frame) / fps
    else:
        duration_rel_s = (last_processed - start_frame + 1) / fps

    total_bins = int(math.ceil(duration_rel_s / args.bin_seconds)) if duration_rel_s>0 else 0
    max_tb     = max(0, total_bins-1)

    if args.no_pad_to_range_end:
        all_tb = (set(tb for tb,_ in counts)
                  | set(streak_only_counts) | set(short_track_counts))
        if all_tb: max_tb = max(all_tb)

    labels = ([f"{edges[i]}_{edges[i+1]}" for i in range(len(edges)-1)]
              + [f"{edges[-1]}_inf"])
    cols = (["time_start_s","time_end_s"]
            + [f"speedbin_{l}" for l in labels]
            + ["streak_only_short","short_track","total_counted"])

    rows = []
    for tb in range(max_tb+1):
        row = [start_s+tb*args.bin_seconds, start_s+(tb+1)*args.bin_seconds]
        tot = 0
        for sb in range(len(edges)-1):
            c=counts.get((tb,sb),0); row.append(c); tot+=c
        c=counts.get((tb,len(edges)-1),0); row.append(c); tot+=c
        cs=streak_only_counts.get(tb,0); ch=short_track_counts.get(tb,0)
        row+=[cs,ch,tot+cs+ch]; rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    sc = [c for c in df.columns if c.startswith("speedbin_")]
    ec = [c for c in ["streak_only_short","short_track"] if c in df.columns]
    df["total_cells"] = df[sc+ec].sum(axis=1)
    df.to_csv(args.out_csv, index=False)
    xl = args.out_csv.replace(".csv",".xlsx")
    df.to_excel(xl, index=False)
    if args.per_object_csv:
        pd.DataFrame(per_rows).to_csv(args.per_object_csv, index=False)
    print("Wrote:", args.out_csv)
    print("Wrote:", xl)
    if args.per_object_csv:
        print("Wrote:", args.per_object_csv)


if __name__ == "__main__":
    main()
