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
            for j, (cx, cy, *_rest) in enumerate(detections):
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
        for j, (cx, cy, *_rest) in enumerate(detections):
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


def find_seeds_prominence(height_map, blob_mask, fg_thresh, prominence_frac, candidate_min_dist):
    """
    Seed-finding that separates two concerns min_peak_dist conflates: "is this a
    distinct peak at all" (candidate_min_dist, kept small) vs "is it tall enough
    relative to its neighbor to really be a separate cell, not noise" (prominence).

    Pure distance-based filtering (peak_local_max with a single min_distance) can't
    do both at once: a distance large enough to reject noise bumps also makes it
    mechanically impossible to split two real cells whose centers are closer
    together than that distance -- common in dense/tightly-clustered regions, where
    a whole multi-cell cluster can be smaller across than the distance threshold
    itself. Prominence instead asks: how much does this peak's height exceed the
    saddle (the lowest point along the ridge) between it and its nearest taller
    neighbor? A real second cell has a deep saddle (its own distinct falloff to
    background); a noise bump on the shoulder of one real cell has a shallow one.

    Finds many close-together candidate peaks first (small candidate_min_dist),
    then greedily merges each into its nearest taller neighbor if the saddle
    between them isn't deep enough (prominence < prominence_frac * height_map.max()).

    height_map is usually the blob's distance transform (a geometric "how far from
    the mask edge" surface), but can also be the raw grayscale image restricted to
    the blob (masked-out pixels set below any real value) -- see use_intensity_peaks
    on split_contours_watershed for why: MOG2's foreground confidence can saturate
    uniformly across two overlapping bright cells even when their actual brightness
    peaks, and the dimmer saddle between them, are still visible in the original
    image. The mask's geometry alone can't recover a boundary that isn't there.
    """
    coords = peak_local_max(height_map, min_distance=max(1, int(candidate_min_dist)),
                             threshold_abs=fg_thresh * height_map.max(), labels=blob_mask)
    if len(coords) <= 1:
        return coords
    heights = height_map[coords[:, 0], coords[:, 1]]
    order = np.argsort(-heights)  # tallest first
    prominence_thresh = prominence_frac * height_map.max()

    kept = []  # (y, x, h), tallest-first order preserved
    for idx in order:
        y, x = coords[idx]
        h = heights[idx]
        merged = False
        for ky, kx, kh in kept:
            n_samples = max(2, int(math.hypot(x - kx, y - ky)))
            xs = np.clip(np.linspace(x, kx, n_samples).astype(int), 0, height_map.shape[1] - 1)
            ys = np.clip(np.linspace(y, ky, n_samples).astype(int), 0, height_map.shape[0] - 1)
            saddle = height_map[ys, xs].min()
            if h - saddle < prominence_thresh:
                merged = True
                break
        if not merged:
            kept.append((y, x, h))
    return np.array([[y, x] for y, x, h in kept])


def merge_seed_coords(coords_a, coords_b, min_dist):
    """
    Union two seed-coordinate arrays (e.g. bright-peak seeds and dark-peak seeds
    from the two intensity polarities), dropping any coords_b point that lands
    within min_dist of an already-kept point -- if both polarity searches landed
    on essentially the same spot, that's one cell, not two.
    """
    if len(coords_a) == 0:
        return coords_b
    if len(coords_b) == 0:
        return coords_a
    kept = list(coords_a)
    for y, x in coords_b:
        if all(math.hypot(x - kx, y - ky) >= min_dist for ky, kx in kept):
            kept.append(np.array([y, x]))
    return np.array(kept)


def estimate_merged_seeds(blob_mask, height_map, expected_count):
    """
    Last-resort fallback for a blob whose area implies more cells than any real
    peak-finding (distance-transform or intensity) could distinguish -- e.g. a
    tightly packed, uniformly saturated cluster where individual cells have
    fully merged with no distinguishable local maximum left anywhere, not even
    on the intensity surface. There's no peak left to find in that case, so
    instead partition the blob's own pixels into expected_count roughly-equal
    regions via k-means on pixel coordinates (this literally does "divide the
    area into one-cell-sized pieces" -- see split_contours_watershed's
    est_cell_area), then snap each region's seed to its own brightest/tallest
    pixel so watershed still floods from a locally sensible starting point.
    Positions from this path are approximate; the goal is only to recover the
    right cell *count* for a box that's otherwise stuck at 1.
    """
    ys, xs = np.nonzero(blob_mask)
    if expected_count >= len(xs):
        expected_count = max(1, len(xs))
    if expected_count <= 1:
        idx = np.argmax(height_map[ys, xs])
        return np.array([[ys[idx], xs[idx]]])
    pts = np.column_stack([xs, ys]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(pts, expected_count, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    seeds = []
    for k in range(expected_count):
        sel = labels == k
        if not sel.any():
            continue
        kys, kxs = ys[sel], xs[sel]
        idx = np.argmax(height_map[kys, kxs])
        seeds.append([kys[idx], kxs[idx]])
    return np.array(seeds)


def find_outer_contours(mask, want_hole_flag=False):
    """
    cv2.findContours with RETR_EXTERNAL, optionally also reporting whether each
    outer contour has an internal hole (a child contour in RETR_CCOMP's
    hierarchy). A hole means the mask is a ring/donut, not a filled disk --
    confirmed on real footage to happen for cells whose interior is dark enough
    to be indistinguishable from the learned MOG2 background (only the bright
    halo rim deviates enough to be flagged foreground), unlike ordinary live
    cells whose whole body reads as foreground. Used to identify that subset
    without a brightness threshold, which swept up far too many normal (dim)
    detections when tried (see --exclude_hole_blobs).
    """
    if not want_hole_flag:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [(c, False) for c in cs]
    cs, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    out = []
    for i, c in enumerate(cs):
        _, _, child, parent = hierarchy[i]
        if parent != -1:
            continue  # this is a hole itself, not a blob -- skip, its parent covers it
        out.append((c, child != -1))
    return out


def split_contours_watershed(binary_mask, frame_bgr, fg_thresh, min_area, min_split_area,
                              min_peak_dist, prominence_frac=None, gray=None,
                              use_intensity_peaks=False, est_cell_area=None,
                              exclude_hole_blobs=False):
    """
    Split touching/overlapping cells using a distance-transform watershed.

    Seeds are normally the distance-transform local maxima (cell centers), found
    with skimage.feature.peak_local_max so that two seeds separated by at least
    min_peak_dist are always kept distinct (a plain dilation-based local-max
    test merges seeds that sit close together, which is exactly the touching-
    cell case this is meant to fix). Flooding uses skimage.segmentation.watershed
    on -dist restricted to the blob mask, so basins sit at the cell centers and
    the split line falls on the true ridge between them. cv2.watershed is not
    used here: its flooding order is driven by local gradient magnitude, which
    is nearly flat across a smooth distance-transform surface, so it produces
    essentially arbitrary (often massively lopsided) splits on this input.

    use_intensity_peaks (needs gray) finds seeds from raw grayscale brightness
    within the blob instead of the mask's distance transform. This matters
    because the mask itself can be the actual bottleneck: MOG2's foreground
    confidence can saturate uniformly across two overlapping bright cells even
    though their real brightness peaks (and the dimmer saddle between them) are
    still visible in the original image -- confirmed directly on real footage,
    where a mask region for a 3+ cell cluster was one undifferentiated blob with
    zero internal structure at any --mask_thresh, while intensity-based peaks on
    the same region found 3 distinct, well-separated, near-saturated maxima. No
    amount of geometric analysis of the mask shape can recover a boundary that
    was already lost when the mask was thresholded. Watershed flooding still
    uses -dist (not intensity) even in this mode, since distance transform
    remains the right surface for drawing clean basin boundaries once seed
    locations are known.
    """
    mask = binary_mask.copy()
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    raw_contours = find_outer_contours(mask, want_hole_flag=True)

    # Returns (contour, seed_xy) pairs. seed_xy is the exact peak pixel for a split
    # cell (the true, accurate cell center); None for a pass-through unsplit contour,
    # where the caller should fall back to the contour's centroid (fine there since
    # it's a single symmetric blob). Using the split *region*'s own centroid instead
    # of its seed is wrong: an uneven/asymmetric split (very common for cells
    # overlapping unevenly) produces a lopsided region whose centroid can land
    # noticeably off the real cell, especially visible for close/touching pairs.
    out = []
    for rc, has_hole in raw_contours:
        area = cv2.contourArea(rc)
        if area < min_area:
            continue
        # A hole only marks a single dead cell if the blob stays a single,
        # unsplit object below -- gating the drop here instead of upfront
        # matters because several live cells clustered around an incidental
        # gap can also produce a mask with a hole, and splitting would still
        # correctly recover them as separate real cells. Excluding upfront
        # would silently drop that whole cluster instead. has_hole itself is
        # always carried through (even when not dropping) so callers can tag
        # a detection as a likely dead cell without discarding it.
        drop_if_unsplit = exclude_hole_blobs and has_hole

        # small blobs → single cell, skip splitting
        if area < min_split_area:
            if not drop_if_unsplit:
                out.append((rc, None, has_hole))
            continue

        # isolate blob
        blob_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [rc], -1, 255, -1)

        dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            if not drop_if_unsplit:
                out.append((rc, None, has_hole))
            continue

        if use_intensity_peaks and gray is not None:
            height_map = np.where(blob_mask > 0, gray.astype(np.float64), -1.0)
            # A single grayscale polarity only finds cells that are brighter than
            # their surroundings. Confirmed on real footage that not every cell in
            # a frame shares the same apparent polarity -- a dark-appearing cell
            # sitting right next to a bright one is a minimum, not a maximum, in
            # `gray`, so it never produces its own peak and the pair collapses
            # into one detection. Searching the inverted map too (dark = high)
            # recovers that second seed; the two seed sets are then merged.
            inv_height_map = np.where(blob_mask > 0, 255.0 - gray.astype(np.float64), -1.0)
        else:
            height_map = dist
            inv_height_map = None

        if prominence_frac is not None:
            coords = find_seeds_prominence(height_map, blob_mask, fg_thresh, prominence_frac,
                                            candidate_min_dist=min_peak_dist)
            if inv_height_map is not None:
                dark_coords = find_seeds_prominence(inv_height_map, blob_mask, fg_thresh,
                                                      prominence_frac, candidate_min_dist=min_peak_dist)
                n_bright = len(coords)
                coords = merge_seed_coords(coords, dark_coords, min_peak_dist)
        else:
            coords = peak_local_max(height_map, min_distance=max(1, int(min_peak_dist)),
                                     threshold_abs=fg_thresh * height_map.max(),
                                     labels=blob_mask)
            if inv_height_map is not None:
                dark_coords = peak_local_max(inv_height_map, min_distance=max(1, int(min_peak_dist)),
                                              threshold_abs=fg_thresh * inv_height_map.max(),
                                              labels=blob_mask)
                n_bright = len(coords)
                coords = merge_seed_coords(coords, dark_coords, min_peak_dist)

        # Bright+dark seed merging can over-generate on a small, noisy/textured
        # blob (confirmed on real footage: a 259px^2 blob produced 7 candidate
        # seeds for what was visibly ~3 cells). Flooding that many seeds into
        # too little area produces mostly slivers below min_area, so every
        # split piece gets dropped and the blob silently falls back to
        # unsplit -- the extra dark seeds need capping by how much area is
        # actually there to split. Never cap below n_bright though: that's
        # exactly what a single-polarity search alone would have kept, and
        # trimming into it would regress below the pre-dark-search behavior,
        # not just tame the new addition.
        if inv_height_map is not None and len(coords) > 1:
            # min_split_area is a floor for "worth attempting a split at all",
            # not a real single-cell size estimate (it's deliberately small --
            # see its default 1.8*min_area), so it under-caps here. A rough
            # single-cell area of ~8*min_area matches the median real contour
            # size measured on this footage (min_area=12 -> ~96px^2, vs. the
            # observed median of ~43-90px^2 across isolated detections).
            max_seeds = max(1, n_bright, round(area / (8 * min_area)))
            if len(coords) > max_seeds:
                coords = coords[:max_seeds]

        if est_cell_area:
            expected_count = max(1, round(area / est_cell_area))
            if expected_count > len(coords):
                coords = estimate_merged_seeds(blob_mask, height_map, expected_count)

        if len(coords) <= 1:
            # only one cell center found → single cell
            if not drop_if_unsplit:
                out.append((rc, None, has_hole))
            continue

        peak_mask = np.zeros(dist.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, n_seeds = ndi.label(peak_mask)

        # map each marker label to the exact peak pixel (row, col) that produced it
        label_to_seed = {}
        for (py, px) in coords:
            label_to_seed[markers[py, px]] = (float(px), float(py))

        labels = sk_watershed(-dist, markers, mask=blob_mask)

        split_any = False
        for label in range(1, n_seeds + 1):
            obj = np.uint8(labels == label) * 255
            cs, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if cv2.contourArea(c) >= min_area:
                    # a piece from an actual multi-seed split is a recovered real
                    # cell, not the dead-cell signature -- never tag it as one
                    out.append((c, label_to_seed.get(label), False))
                    split_any = True

        if not split_any and not drop_if_unsplit:
            out.append((rc, None, has_hole))

    return out


def local_blob_sharpness(gray, cx, cy, radius):
    """
    Laplacian variance (a standard focus/blur metric) in a small patch around
    (cx, cy) on an unblurred grayscale image -- low variance = few sharp edges
    = blurry. This is the dead-cell signature confirmed directly against
    user-labeled examples (t=280s, detections 143/13/17/43/113/118/34): unlike
    the sharp, high-contrast bright-halo ring of a live cell, these show up as
    a soft, blurred, halo-less dark blob -- consistent with sitting outside the
    imaging focal plane (e.g. settled to the channel floor) rather than flowing
    through it. Replaces an earlier mask-hole-based guess the user directly
    confirmed was wrong (none of the 7 labeled examples had a mask hole).

    Returns None if the patch would be clipped by the frame border -- confirmed
    to read as artificially uniform/low-variance there too, which would wrongly
    flag real edge-of-frame cells as dead.

    NOTE: the raw variance's absolute scale does drift some between frames
    (confirmed: an earlier, lower-density timepoint measured a lower median
    than a later, denser one), which is why --dead_cell_max_sharpness needed
    retuning (550 -> 450) once checked against a second timepoint. A per-frame
    relative/percentile threshold was tried as a fix but rejected: the user
    confirmed a frame can legitimately be mostly dead, which a percentile rank
    can never express since it always tags close to the requested percentile
    regardless of the true proportion. See --dead_cell_max_sharpness.
    """
    y0, y1 = cy - radius, cy + radius
    x0, x1 = cx - radius, cx + radius
    if y0 < 0 or x0 < 0 or y1 > gray.shape[0] or x1 > gray.shape[1]:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float64)
    return cv2.Laplacian(patch, cv2.CV_64F).var()


def calibrate_dead_cell_reference(video_path, warmup_start, start_frame, fps, args, percentile=75,
                                   min_samples=200, max_sample_s=120):
    """
    --dead_cell_max_sharpness is a fixed pixel-variance number, which only means
    anything relative to one specific video's own brightness/contrast -- confirmed
    directly: a value (450) validated on one video wrongly tagged ~84% of cells as
    dead on a second, visually darker/lower-contrast video, because that video's
    whole sharpness scale sits lower across the board, not because most of its
    cells are actually dead. Rather than requiring a fresh labeled example and a
    manual retune for every new video, this measures THIS video's own typical
    sharpness once (a throwaway MOG2 pass over the start of the analysis window,
    not reused for the real run -- needs its own instance since MOG2 is stateful)
    and returns a percentile of it, so --dead_cell_relative_thresh (a fraction of
    this reference) can set an absolute cutoff that auto-scales per video instead
    of a hardcoded number tuned on different footage.

    Deliberately samples starting at start_frame (not warmup_start): the
    background model is still adapting during warmup, which would bias the
    measured sharpness distribution.

    A fixed 10s sample window is only reliable if that window happens to be
    busy -- confirmed on real footage to fail exactly this way: a video whose
    first ~90s were a near-empty priming/pre-flow period (0-4 detections per
    frame, same pattern seen on other videos from this setup) gave a
    calibration reference built from only a handful of samples, most of them
    the same one or two cells re-measured across consecutive frames rather
    than independent cells -- not a reliable estimate of the video's real
    sharpness distribution. Extends the sampling window (up to
    max_sample_s) until at least min_samples independent-ish measurements
    are collected, since a low-activity start is common in this kind of
    flow assay, not an edge case.

    Returns (reference_value, n_samples, sampled_s) or (None, 0, sampled_s)
    if nothing was found in the whole extended window.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=args.mog2_history, varThreshold=args.mog2_varThreshold, detectShadows=False)
    sample_end = start_frame + int(max_sample_s * fps)
    sharpness_vals = []
    f = warmup_start
    while f < sample_end:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.gauss_ksize > 0:
            k = args.gauss_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        fgm = backsub.apply(gray, learningRate=float(args.learning_rate))
        _, th = cv2.threshold(fgm, args.mask_thresh, 255, cv2.THRESH_BINARY)
        if f >= start_frame:
            sharp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                area = cv2.contourArea(c)
                if area < args.min_area or area > args.max_area:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                sv = local_blob_sharpness(sharp_gray, cx, cy, args.dead_cell_sharpness_radius)
                if sv is not None:
                    sharpness_vals.append(sv)
            if len(sharpness_vals) >= min_samples and f >= start_frame + int(10 * fps):
                break
        f += 1
    cap.release()
    sampled_s = (min(f, sample_end) - start_frame) / fps
    if not sharpness_vals:
        return None, 0, sampled_s
    return float(np.percentile(sharpness_vals, percentile)), len(sharpness_vals), sampled_s


def calibrate_mog2_varThreshold(video_path, warmup_start, start_frame, fps, args,
                                 candidates=(16.0, 8.0, 4.0, 2.0, 1.0, 0.5),
                                 oversized_area_mult=40, max_oversized_frac=0.06,
                                 max_growth_ratio=2.0, growth_smooth_k=3,
                                 max_aspect_ratio=4.0, max_sliver_frac=0.10,
                                 sample_s=10.0, n_evals=5):
    """
    --mog2_varThreshold is a fixed absolute cutoff, and confirmed directly on
    real footage that the right value is NOT a per-video constant, let alone a
    global one: the same video's low-contrast region (reference sharpness ~17,
    see --dead_cell_relative_thresh) needed varThreshold ~1 to detect real
    cells that a high-contrast region of the same video (reference sharpness
    ~805) detected fine at varThreshold~4. Lowering varThreshold to recover
    faint cells always comes at a cost though, and confirmed on real footage
    that it's not just one cost but three independent ones, so this checks
    for all three:

    1. Merging: every detected blob's mask gets puffier as varThreshold drops,
       making touching neighbors more likely to merge into one oversized box
       (t=100s: varThreshold 4->1 raised detection count 264->294, but raised
       oversized-box count 12->47, a much bigger relative jump). Caught via
       the fraction of boxes with area > oversized_area_mult * min_area.
    2. Sudden noise onset: below some threshold, MOG2 starts flagging
       flat/featureless background as foreground, producing lots of small
       non-oversized boxes that (1) can't catch since they're not merged,
       they're just wrong. Confirmed directly: on a genuinely sparse region,
       oversized_frac stayed at 0.00 all the way down to varThreshold=0.5
       even though that setting visually put boxes on plain background with
       no cell in them at all -- count jumped 17->89 with nothing to show for
       it. A real cell population grows smoothly as sensitivity increases
       (t=100s ratios stayed <=1.8x per halving all the way to 0.5); noise
       onset instead shows a sudden jump (the same sparse region jumped 2.5x,
       then 4.6x, right at the point confirmed visually to already be noise)
       -- caught via a smoothed growth-ratio cap between consecutive
       candidates ((n_new+k)/(n_old+k), the +k damping ratio instability when
       counts are still tiny).
    3. Sliver artifacts one or two at a time: distinct from (2) -- these
       don't show up as a sudden count jump because only 1-2 appear per
       candidate step, but they're still noise, not cells. Confirmed directly
       (t=850s, varThreshold 16->2): the 3 real cells at varThreshold=16 were
       still correctly found at varThreshold=2, but alongside 2 new
       detections shaped (30x3) and (2x23) px -- extreme aspect ratios no
       round cell produces. Neither the area check (not oversized) nor the
       growth-ratio check (3->5 isn't a sudden jump) catches this. Caught via
       the fraction of boxes whose long/short side ratio exceeds
       max_aspect_ratio.

    A candidate must pass all three checks to be accepted; sweeps high->low
    so the lowest (most sensitive) passing candidate wins.

    Deliberately re-decodes+reruns MOG2 once per candidate rather than sharing
    one decoded-frame cache across all of them: keeps this self-contained and
    match calibrate_dead_cell_reference's "own instance, MOG2 is stateful"
    reasoning, at the cost of a few seconds of extra one-time calibration work
    -- the same tradeoff already accepted there.

    Evaluates detect_cells() on every stride-th frame from start_frame through
    start_frame+sample_s (not just once at the end of that window) -- a single
    evaluation frame turned out to be unreliable (confirmed directly: a
    candidate that looked fine at its window's last frame looked clearly
    over-merged when independently checked at start_frame itself, because
    MOG2's model keeps evolving through the sample window, so the endpoint
    frame doesn't represent the frame this calibration is actually supposed
    to be representative of). The oversized/sliver fractions are taken as the
    WORST (max) across evaluated frames, not pooled/averaged together --
    confirmed directly that averaging lets a bad frame's problem get diluted
    by cleaner frames elsewhere in the same sample window (sliver_frac 0.29 at
    the actual target frame vs. a pooled 0.06 across the whole window), which
    would silently pass a candidate that's bad exactly where it matters.
    n_detections is still summed across frames (just a rough total, not a
    pass/fail signal).

    Returns (best_varThreshold, stats_by_candidate) where stats_by_candidate
    maps each tried value to (n_detections, oversized_frac, sliver_frac,
    growth_ratio), for the caller to print/log.
    """
    # Tried capping this to a shorter throwaway warmup to cut the 6x redundant-warmup cost
    # that makes this slow -- confirmed directly that's NOT safe: MOG2's background model
    # genuinely hasn't finished settling at 15s vs. 45s, and it changed the actual answer
    # (t=850s, varThreshold=2: 15s-warmup found 6 clean boxes, 45s-warmup found 7 including
    # 2 real sliver artifacts the shorter warmup simply hadn't learned to suppress yet). A
    # calibration that doesn't match the warmup the real run will actually have at that
    # point isn't testing the right thing, so this always uses the caller's full
    # warmup_start, at the cost of the 6x-redundant-warmup slowness that comes with it.
    sample_end = start_frame + int(sample_s * fps)
    eval_stride = max(1, int(sample_s * fps) // n_evals)
    # A candidate found with zero (or almost zero) real detections trivially passes every
    # check below -- an empty sample has no merging, no noise jump, no slivers to measure.
    # Confirmed this is a real failure mode, not hypothetical: a calibration window landing
    # in this video's known low-activity start found 0 detections at varThreshold 16/8/4/2
    # and picked 2 as "clean" with zero actual evidence it's any better than 16, only
    # stopping once varThreshold=1 finally produced 5 detections that were 100% slivers.
    # Candidates below this floor are treated as unproven, not passing.
    min_detections_to_trust = 15
    stats = {}
    best = candidates[0]  # fall back to the most conservative (highest) if none pass/qualify
    prev_n = None
    for varT in candidates:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
        backsub = cv2.createBackgroundSubtractorMOG2(
            history=args.mog2_history, varThreshold=varT, detectShadows=False)
        f = warmup_start
        total_n = 0
        frame_fracs = []  # (oversized_frac, sliver_frac) per evaluated frame -- worst-case
                           # across frames is what gates a candidate, not the pooled average
                           # (confirmed: pooling let a bad frame's high sliver_frac get diluted
                           # by cleaner frames elsewhere in the sample window, masking a real
                           # problem at the specific frame --start_s actually starts counting at)
        while f < sample_end:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.gauss_ksize > 0:
                k = args.gauss_ksize | 1
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            fgm = backsub.apply(gray, learningRate=float(args.learning_rate))
            _, th = cv2.threshold(fgm, args.mask_thresh, 255, cv2.THRESH_BINARY)
            if f >= start_frame and (f - start_frame) % eval_stride == 0:
                _, _, boxes = detect_cells(th, frame, gray, args, fg_thresh=args.watershed_fg_thresh,
                                            min_peak_dist=args.watershed_min_peak_dist,
                                            prominence_frac=args.watershed_prominence_frac)
                n = len(boxes)
                total_n += n
                oversized = sum(1 for (x, y, w, h, is_s, is_dead) in boxes
                                 if w * h > oversized_area_mult * args.min_area)
                sliver = sum(1 for (x, y, w, h, is_s, is_dead) in boxes
                             if max(w, h) > max_aspect_ratio * max(1, min(w, h)))
                frame_fracs.append((oversized / n if n else 0.0, sliver / n if n else 0.0))
            f += 1
        cap.release()
        oversized_frac = max((of for of, sf in frame_fracs), default=0.0)
        sliver_frac = max((sf for of, sf in frame_fracs), default=0.0)
        growth = ((total_n + growth_smooth_k) / (prev_n + growth_smooth_k)
                  if prev_n is not None else 1.0)
        stats[varT] = (total_n, oversized_frac, sliver_frac, growth)
        if (total_n >= min_detections_to_trust
                and oversized_frac <= max_oversized_frac and sliver_frac <= max_sliver_frac
                and growth < max_growth_ratio):
            best = varT  # candidates iterate high->low, so last passing one is lowest/most sensitive
        prev_n = total_n

    if stats.get(best, (0,))[0] < min_detections_to_trust:
        print(f"WARNING: --auto_mog2_varThreshold calibration found fewer than "
              f"{min_detections_to_trust} detections at every candidate threshold in this "
              f"{sample_s:.0f}s sample window -- this looks like a low-activity period, not a "
              f"clean-vs-noisy tradeoff. Falling back to the conservative default "
              f"({candidates[0]:g}) rather than trusting a threshold with no real evidence "
              f"behind it. If this run's counts look too low, consider a later --start_s or "
              f"passing --mog2_varThreshold manually.")
        best = candidates[0]

    return best, stats


def detect_cells(th, frame, gray, args, fg_thresh=None, min_peak_dist=None, prominence_frac=None,
                  raw_diff=None):
    """
    Mask -> contours -> filtered (cx, cy, is_streak) detections. Shared by the
    real per-frame loop and the quick preview/sweep path so both always see
    identical detection behavior. fg_thresh / min_peak_dist / prominence_frac let
    the preview path override the CLI values per-tile without touching args.
    """
    if args.use_watershed_split:
        min_split = args.watershed_min_split_area or (1.8 * args.min_area)
        if fg_thresh is None:
            fg_thresh = args.watershed_fg_thresh
        if min_peak_dist is None:
            min_peak_dist = args.watershed_min_peak_dist or math.sqrt(min_split / (2 * math.pi))
        if prominence_frac is None:
            prominence_frac = args.watershed_prominence_frac
        contour_seed_pairs = split_contours_watershed(th, frame, fg_thresh, args.min_area,
                                            min_split_area=min_split,
                                            min_peak_dist=min_peak_dist,
                                            prominence_frac=prominence_frac,
                                            gray=gray,
                                            use_intensity_peaks=args.watershed_use_intensity,
                                            est_cell_area=args.watershed_est_cell_area,
                                            exclude_hole_blobs=args.exclude_hole_blobs)
    else:
        cs_holes = find_outer_contours(th, want_hole_flag=True)
        contour_seed_pairs = [(c, None, has_hole) for c, has_hole in cs_holes
                               if not (args.exclude_hole_blobs and has_hole)]

    # gray is Gaussian-blurred upstream (for MOG2/mask purposes) -- that smoothing
    # destroys exactly the high-frequency edge content the blur/sharpness dead-cell
    # signal depends on (confirmed: measuring on the blurred gray reads almost
    # every detection as "blurry", not just the true dead cells). Recompute an
    # unblurred grayscale from the raw frame instead, once per call, only when
    # actually needed.
    want_sharpness = args.dead_cell_max_sharpness is not None or args.dead_cell_percentile is not None
    sharp_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if want_sharpness else None

    # Staged first pass: gather everything except the dead-cell tag, since
    # --dead_cell_percentile needs every candidate's sharpness collected before
    # it can rank them against each other (see below).
    staged = []
    for c, seed, _has_hole in contour_seed_pairs:
        area = cv2.contourArea(c)
        if area < args.min_area or area > args.max_area:
            continue
        if args.min_mean_intensity > 0:
            blob_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=blob_mask)[0]
            if mean_intensity < args.min_mean_intensity:
                continue  # dark/black blob (e.g. debris, dead cell) - not counted
        if seed is not None:
            # true distance-transform peak for a split cell -- accurate even when
            # the split region itself is lopsided/asymmetric
            cx, cy = int(round(seed[0])), int(round(seed[1]))
        else:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        if args.min_local_motion > 0 and raw_diff is not None:
            r = 4
            local = raw_diff[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
            if local.size == 0 or local.mean() < args.min_local_motion:
                # No real frame-to-frame change here -- a MOG2 "ghost" (something,
                # often debris, was recently here but has already moved on; the mask
                # still flags it from residual variance even though current pixels
                # look like ordinary background) rather than a currently-present cell.
                continue
        x, y, w, h = cv2.boundingRect(c)
        long_axis  = float(max(w, h))
        short_axis = float(max(1, min(w, h)))
        is_streak  = (args.enable_streak
                      and area >= args.streak_min_area
                      and long_axis >= args.streak_min_len
                      and long_axis/short_axis >= args.streak_ar)
        sharpness = local_blob_sharpness(sharp_gray, cx, cy, args.dead_cell_sharpness_radius) \
                    if sharp_gray is not None else None
        staged.append((cx, cy, is_streak, x, y, w, h, sharpness))

    # --dead_cell_percentile ranks each detection against this frame's own
    # sharpness distribution instead of a fixed absolute cutoff -- confirmed
    # necessary on real footage: a whole frame's baseline sharpness can differ
    # enough between timepoints (an earlier, lower-density frame measured
    # roughly half the median variance of a later, denser one) that a single
    # fixed --dead_cell_max_sharpness value either over-flagged real live cells
    # (sparse/dim frame) or under-flagged real dead ones (dense/sharp frame).
    percentile_cutoff = None
    if args.dead_cell_percentile is not None:
        valid = [s[7] for s in staged if s[7] is not None]
        if valid:
            percentile_cutoff = np.percentile(valid, args.dead_cell_percentile)

    detections = []
    det_boxes  = []
    for cx, cy, is_streak, x, y, w, h, sharpness in staged:
        is_dead = False
        if sharpness is not None:
            if percentile_cutoff is not None:
                is_dead = sharpness < percentile_cutoff
            elif args.dead_cell_max_sharpness is not None:
                is_dead = sharpness < args.dead_cell_max_sharpness
        detections.append((cx, cy, is_streak, is_dead))
        det_boxes.append((x, y, w, h, is_streak, is_dead))
    return [c for c, _, _ in contour_seed_pairs], detections, det_boxes


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


def save_preview(th, frame, gray, args, raw_diff=None):
    """
    Annotate one already-computed frame/mask with detected cell counts and
    save it, instead of writing a full-video debug file. If --sweep_fg_thresh
    and/or --sweep_min_peak_dist give more than one value, builds a grid with
    one tile per combination (rows = min_peak_dist, cols = fg_thresh) so
    several settings can be compared at a glance from a single frame.

    If --sweep_prominence_frac is given, it replaces --sweep_min_peak_dist as
    the row axis (prominence and min_peak_dist both control split sensitivity,
    so sweeping both at once isn't useful) -- --watershed_min_peak_dist is then
    held fixed as the candidate-peak spacing for every tile.
    """
    print_blob_size_diagnostics(th, args)

    # Save the exact, untouched frame this preview was built from, from the same
    # in-memory frame object used for detection -- so a raw-vs-detected comparison
    # is guaranteed pixel-identical instead of relying on separately re-deriving
    # which frame index to re-extract (which has repeatedly gone wrong: off-by-one
    # errors between a manual reproduction's target-frame math and this tool's
    # actual >=-based trigger condition).
    raw_out = args.preview_out.rsplit(".", 1)
    raw_out = f"{raw_out[0]}_raw.{raw_out[1]}" if len(raw_out) == 2 else args.preview_out + "_raw"
    cv2.imwrite(raw_out, frame)
    print(f"Wrote: {raw_out}  (raw frame, no annotations, for direct before/after comparison)")

    # Same idea as _raw above, but for the actual binary foreground mask -- lets
    # you tell apart "MOG2 never flagged this object as foreground at all" (not
    # in this image either) from "it was in the mask but got filtered/merged
    # away downstream" (visible here, missing from the annotated output).
    mask_out = args.preview_out.rsplit(".", 1)
    mask_out = f"{mask_out[0]}_mask.{mask_out[1]}" if len(mask_out) == 2 else args.preview_out + "_mask"
    cv2.imwrite(mask_out, th)
    print(f"Wrote: {mask_out}  (binary foreground mask, same frame)")

    if not args.use_watershed_split and (args.sweep_fg_thresh or args.sweep_min_peak_dist
                                          or args.sweep_prominence_frac):
        print("Note: --sweep_fg_thresh/--sweep_min_peak_dist/--sweep_prominence_frac only "
              "affect anything when --use_watershed_split is also passed.")

    fg_list = ([float(x) for x in args.sweep_fg_thresh.split(",")] if args.sweep_fg_thresh
               else [args.watershed_fg_thresh])

    if args.sweep_prominence_frac:
        row_label = "prominence_frac"
        row_list = [float(x) for x in args.sweep_prominence_frac.split(",")]
        fixed_pd = args.watershed_min_peak_dist or 4
        def make_kwargs(row_val):
            return dict(min_peak_dist=fixed_pd, prominence_frac=row_val)
    elif args.sweep_min_peak_dist:
        row_label = "min_peak_dist"
        row_list = [float(x) for x in args.sweep_min_peak_dist.split(",")]
        def make_kwargs(row_val):
            return dict(min_peak_dist=row_val, prominence_frac=args.watershed_prominence_frac)
    else:
        row_label = "min_peak_dist"
        min_split = args.watershed_min_split_area or (1.8 * args.min_area)
        row_list = [args.watershed_min_peak_dist or math.sqrt(min_split / (2 * math.pi))]
        def make_kwargs(row_val):
            return dict(min_peak_dist=row_val, prominence_frac=args.watershed_prominence_frac)

    print(f"{'fg_thresh':>10} {row_label:>16} {'count':>6}")
    rows = []
    for row_val in row_list:
        tiles = []
        for fg in fg_list:
            _, detections, _ = detect_cells(th, frame, gray, args, fg_thresh=fg,
                                             raw_diff=raw_diff, **make_kwargs(row_val))
            print(f"{fg:>10.2f} {row_val:>16.2f} {len(detections):>6}")
            tile = frame.copy()
            for i, (cx, cy, _is_streak, is_dead) in enumerate(detections):
                col = (0, 0, 255) if is_dead else (0, 255, 0)
                cv2.circle(tile, (cx, cy), 3, col, -1)
                cv2.putText(tile, str(i + 1), (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(tile, f"fg={fg:g} {row_label[:4]}={row_val:g} n={len(detections)}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    cv2.imwrite(args.preview_out, grid)
    print(f"Wrote: {args.preview_out}  ({len(row_list)}x{len(fg_list)} grid)")


def save_crossing_contact_sheet(video_path, per_rows, fps, out_path, crop=120, max_cols=10):
    """
    One small thumbnail per counted cell, cropped from the real frame at the
    moment it was counted (time_exit_s_abs/x_exit/y_exit from per_rows) and
    centered with a marker. Lets someone verify precision (is each count a
    real cell, not noise/debris) by glancing at a single image instead of
    manually tallying crossings while watching the full video -- which isn't
    practical for a fast, dense clip. Does not verify recall (missed cells
    won't show up here since there's nothing to crop them from).
    """
    if not per_rows:
        print("No counted cells to build a contact sheet from.")
        return

    cap2 = cv2.VideoCapture(video_path)
    if not cap2.isOpened():
        print(f"Could not reopen {video_path} to build contact sheet.")
        return

    # cv2's frame-index seeking (CAP_PROP_POS_FRAMES) is unreliable on many
    # compressed formats -- it can land near the nearest keyframe rather than the
    # exact requested frame, silently cropping the wrong moment. An earlier version
    # of this function still seeked once to a "safe" starting point before reading
    # sequentially, trusting that seek to be accurate -- but if that one seek is off,
    # every frame index after it is offset by the same amount, silently misaligning
    # every thumbnail in the whole sheet at once (confirmed: a contact sheet built
    # this way had zero correctly-aligned thumbnails, while frames read sequentially
    # from frame 0 with no seek at all were correct). So: no seeking at all, ever --
    # always read sequentially from frame 0, which is the only guaranteed-accurate
    # starting point regardless of codec/keyframe interval.
    needed = sorted({int(round(row["time_exit_s_abs"] * fps)) for row in per_rows})
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_by_idx = {}
    cur, ni = 0, 0
    while ni < len(needed):
        ret, frame = cap2.read()
        if not ret:
            break
        if cur == needed[ni]:
            frame_by_idx[cur] = frame.copy()
            ni += 1
        cur += 1
    cap2.release()

    thumbs = []
    n_missing = 0
    for row in per_rows:
        frame_idx = int(round(row["time_exit_s_abs"] * fps))
        frame = frame_by_idx.get(frame_idx)
        if frame is None:
            n_missing += 1
            continue
        H, W = frame.shape[:2]
        x, y = int(row["x_exit"]), int(row["y_exit"])
        half = crop // 2
        # Crop at a fixed window [x-half, x+half) x [y-half, y+half) around the true
        # position and paste onto a crop x crop canvas at 1:1 scale -- do NOT resize
        # an edge-clipped (asymmetric) crop back up to a fixed size, since that
        # stretches the image and silently shifts where the true position actually
        # lands relative to a marker drawn at a fixed center. Every exit near a frame
        # boundary (which is all of them here, since e.g. exit_side=top means y_exit
        # is always small) would otherwise be systematically misaligned. Anything
        # outside the frame is just left black.
        x0, x1 = x - half, x + half
        y0, y1 = y - half, y + half
        src_x0, src_x1 = max(0, x0), min(W, x1)
        src_y0, src_y1 = max(0, y0), min(H, y1)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            n_missing += 1
            continue
        thumb = np.zeros((crop, crop, 3), dtype=np.uint8)
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)
        thumb[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        cv2.circle(thumb, (half, half), 3, (0, 255, 0), 1)
        counted_as = row.get("counted_as", "speedbin")
        # yellow = counted in a speed bin (part of total_counted); orange = only in
        # total_cells via short_track/streak_only_short -- the ones most worth a
        # second look, since they were only tracked a frame or two.
        color = (0, 255, 255) if counted_as == "speedbin" else (0, 140, 255)
        # Exact timestamp + position printed on every thumbnail so this can be checked
        # against the original video independently, in any ordinary video player, with
        # no dependency on this tool's own cropping/rendering being correct.
        label1 = f"id{row['track_id']} {row['final_reason'][:4]}"
        label2 = f"t={row['time_exit_s_abs']:.3f}s ({x},{y})"
        cv2.putText(thumb, label1, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(thumb, label2, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        thumbs.append(thumb)
    if n_missing:
        print(f"Note: {n_missing} row(s) could not be re-extracted.")

    if not thumbs:
        print("Could not extract any crossing thumbnails (video re-seek failed).")
        return

    cols = min(max_cols, len(thumbs))
    n_rows = math.ceil(len(thumbs) / cols)
    pad = cols * n_rows - len(thumbs)
    thumbs += [np.zeros((crop, crop, 3), np.uint8)] * pad
    grid = np.vstack([np.hstack(thumbs[i * cols:(i + 1) * cols]) for i in range(n_rows)])
    cv2.imwrite(out_path, grid)
    print(f"Wrote: {out_path} ({len(thumbs) - pad}/{len(per_rows)} counted-cell thumbnails, "
          f"{cols}x{n_rows} grid)")


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
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Only fully process 1 out of every N frames (default 1 = every "
                         "frame); the other N-1 are cheaply skipped via cap.grab() (no "
                         "decode, no MOG2, no tracking), cutting runtime roughly N-fold on "
                         "high-fps footage that oversamples relative to how fast cells "
                         "actually move. Frame indices (and therefore all time/speed math) "
                         "stay in true video time -- only the gap between frames the "
                         "tracker actually sees grows, so --max_dist likely needs "
                         "increasing by roughly the same factor N or fast cells will "
                         "fragment into multiple short tracks instead of one. Verify cell "
                         "counts are stable before/after enabling this (--preview_frame_s) "
                         "rather than assuming it's free.")

    ap.add_argument("--min_area",   type=float, default=12)
    ap.add_argument("--max_area",   type=float, default=8000)
    ap.add_argument("--max_dist",   type=float, default=220)
    ap.add_argument("--max_missed", type=int,   default=20)
    ap.add_argument("--min_mean_intensity", type=float, default=0.0,
                    help="Minimum mean grayscale intensity (0-255) inside a detected blob "
                         "for it to be counted as a cell. Blobs darker than this ('black' "
                         "cells, debris, dead cells) are discarded before tracking. "
                         "Default 0 = no filtering. Checked on real footage and deliberately "
                         "left off: confirmed solid dark halo-less blobs (debris/stuck "
                         "objects) do get counted, but a threshold strong enough to exclude "
                         "them also caught ~40% of all detections in the same dense frame, "
                         "most of which looked like faint/dim real detections rather than "
                         "debris -- same false-negative risk as --min_local_motion below, and "
                         "not worth it given this assay specifically cares about slower/dimmer "
                         "cells. Use --preview_frame_s to check a value against a real frame "
                         "before committing to a full run if you want to revisit this.")
    ap.add_argument("--exclude_hole_blobs", action="store_true",
                    help="Exclude blobs whose foreground mask has a hole (a donut/ring shape: "
                         "solid bright rim, unflagged interior) instead of being a filled disk. "
                         "NOTE: this was originally built as a dead-cell detector but the user "
                         "directly confirmed on real labeled examples that it's wrong for that -- "
                         "none of 7 user-identified dead cells at t=280s had a mask hole. Kept as "
                         "a narrow, independent shape/topology filter (still legitimately finds "
                         "blobs with an unflagged interior), just no longer tied to dead-cell "
                         "classification. See --dead_cell_max_sharpness for the actual dead-cell "
                         "signal. Off by default.")
    ap.add_argument("--dead_cell_max_sharpness", type=float, default=None,
                    help="Tag a detection as a dead cell if the Laplacian variance (a standard "
                         "focus/blur metric), measured on the raw unblurred frame (not the "
                         "Gaussian-smoothed gray used for MOG2 -- that smoothing was confirmed to "
                         "wash out this signal almost entirely, reading nearly everything as "
                         "'blurry'), in a small patch around its center is below this FIXED value. "
                         "This is the recommended mode -- --dead_cell_percentile (below) was tried "
                         "first but rejected: it ranks within each frame and so mechanically forces "
                         "a fixed tag rate everywhere, which cannot represent a real frame where "
                         "most cells actually are dead (confirmed: the user identified t=90s as "
                         "mostly-dead-except-two, which a percentile-based rank can never output). "
                         "Try 450 as a starting point for this footage -- confirmed against 7 "
                         "user-labeled dead cells (t=280s, 5/7 tagged) and 2 user-confirmed-alive "
                         "cells at t=90s that a higher value (550) wrongly caught (both correctly "
                         "excluded at 450, while still tagging ~75%% of that frame as dead, matching "
                         "the user's own read of it). Tags only (see is_dead_cell in "
                         "--per_object_csv and dead_cell_count in --out_csv); does not exclude "
                         "anything from the count. Default None = disabled. Ignored if "
                         "--dead_cell_percentile is also set.")
    ap.add_argument("--dead_cell_percentile", type=float, default=None,
                    help="Tag a detection as a dead cell if its Laplacian variance (see "
                         "--dead_cell_max_sharpness for what this measures and why) falls below "
                         "this percentile (0-100) of all detections in the SAME frame, instead of "
                         "a fixed absolute value. NOT recommended -- tried this to fix "
                         "--dead_cell_max_sharpness's cross-frame drift problem, but the user then "
                         "confirmed a frame can legitimately be mostly-dead (t=90s, ~75%% dead by "
                         "their own read), which ranking within the frame can never express since "
                         "it always tags close to the requested percentile regardless of the true "
                         "proportion. Kept available in case a use case genuinely wants relative "
                         "ranking, but --dead_cell_max_sharpness is the validated default choice. "
                         "Tags only, does not exclude anything. Default None = disabled.")
    ap.add_argument("--dead_cell_sharpness_radius", type=int, default=8,
                    help="Patch half-size (px) for the dead-cell Laplacian variance measurement "
                         "(--dead_cell_max_sharpness / --dead_cell_percentile). Default 8, matched "
                         "to this footage's cell size -- a patch much smaller than one cell mixes "
                         "in background noise, much larger mixes in neighboring cells (confirmed "
                         "to inflate the score and mask the signal on a cell sitting next to a "
                         "bright neighbor).")
    ap.add_argument("--dead_cell_relative_thresh", type=float, default=None,
                    help="Auto-calibrating alternative to --dead_cell_max_sharpness: instead of a "
                         "fixed pixel-variance number (which only means something relative to the "
                         "specific video it was tuned on -- confirmed a value validated on one "
                         "video wrongly tagged ~84%% of cells as dead on a second, visually darker "
                         "video, since that video's whole sharpness scale sits lower across the "
                         "board), this measures THIS video's own typical in-focus sharpness once "
                         "(a throwaway pass over the first ~10s of the analysis window) and sets "
                         "the actual cutoff to this fraction of that reference. Try 0.4-0.45 as a "
                         "starting point (back-derived from the 450 cutoff that worked against a "
                         "video whose own reference sharpness measured ~1030). Overrides "
                         "--dead_cell_max_sharpness if both are set. Still tags only, does not "
                         "exclude anything from the count -- and still worth spot-checking with "
                         "--preview_frame_s on any new video rather than trusting blindly.")
    ap.add_argument("--min_local_motion", type=float, default=0.0,
                    help="Minimum mean frame-to-frame pixel difference (0-255) in a small "
                         "window around a detected cell's position for it to count. Rejects "
                         "MOG2 'ghosts': a spot flagged as foreground because something "
                         "(often debris) was recently there but has since moved on -- current "
                         "pixels there look like ordinary background, so --min_mean_intensity "
                         "can't catch it. CAUTION: confirmed via direct multi-frame check to "
                         "also reject real, slow-moving cells -- a genuinely present, moving "
                         "cell can still have near-zero displacement between one particular "
                         "pair of adjacent frames just by chance, and one frame-pair's "
                         "instantaneous diff can't tell that apart from a truly static ghost. "
                         "Do not use this if slow-moving cells are part of what you're trying "
                         "to detect (e.g. distinguishing free-flow vs. adhesion-interacted "
                         "cells by speed) -- it will bias against exactly that population. "
                         "Default 0 = no filtering, recommended.")

    ap.add_argument("--mog2_history",      type=int,   default=500)
    ap.add_argument("--mog2_varThreshold", type=float, default=16)
    ap.add_argument("--auto_mog2_varThreshold", action="store_true",
                    help="Auto-calibrate --mog2_varThreshold instead of using a fixed number "
                         "(overrides --mog2_varThreshold if both are given). Confirmed the right "
                         "value is not even a per-video constant, let alone a global default -- "
                         "it needs to track how faint real cells are against the background in "
                         "this specific run's footage. Sweeps a few candidate values on a "
                         "throwaway calibration pass and picks the most sensitive one (best at "
                         "catching faint cells) that doesn't push too many detections into "
                         "oversized/likely-merged boxes. See calibrate_mog2_varThreshold().")
    ap.add_argument("--recalibrate_varThreshold_every_s", type=float, default=120.0,
                    help="With --auto_mog2_varThreshold, re-run the calibration sweep every N "
                         "seconds of video during the real run and update the live MOG2 "
                         "instance's threshold in place (cv2's setVarThreshold(), which does not "
                         "reset its accumulated background model). A single calibration at the "
                         "start of the run is not enough: confirmed the right value drifts within "
                         "a single video as density/contrast changes over time, not just between "
                         "videos, so a threshold picked from the first 10s and used for the whole "
                         "run is wrong everywhere else the video looks different. Set to 0 to "
                         "calibrate once at the start only. Default 120s.")
    ap.add_argument("--varThreshold_schedule", type=str, default=None,
                    help="Cheaper alternative to --auto_mog2_varThreshold: a fixed, hand-picked "
                         "schedule instead of re-running the calibration sweep (6 candidates x "
                         "5 eval frames each, repeated every --recalibrate_varThreshold_every_s) "
                         "-- that repeated cost is real and this skips it entirely, at the cost "
                         "of not adapting to variation the schedule's breakpoints don't happen "
                         "to land on. Format: 'time_s:value,time_s:value,...' e.g. "
                         "'0:0.5,300:4' -- varThreshold=0.5 from t=0, switching to 4 at t=300s. "
                         "Applied via setVarThreshold() (does not reset the background model). "
                         "Overrides --auto_mog2_varThreshold if both are given.")
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
    ap.add_argument("--watershed_min_peak_dist", type=float, default=4.0,
                    help="Minimum distance (px) between two cell-center seeds for them to "
                         "be split into separate cells. Two touching cells closer together "
                         "than this are kept as one. Default 4: the candidate-peak spacing "
                         "confirmed on real footage together with --watershed_prominence_frac "
                         "0.25 (also the default), which handles rejecting noise so this can "
                         "stay small. Pass 0 to fall back to the old radius-based auto-derive "
                         "(sqrt(min_split_area / (2*pi))) instead, which was never validated "
                         "against real footage and produced a much smaller (more aggressive, "
                         "untested) value in practice.")
    ap.add_argument("--watershed_prominence_frac", type=float, default=0.25,
                    help="Enables prominence-based seed filtering instead of pure distance: "
                         "a candidate peak is kept only if it stands at least this fraction "
                         "of the blob's own max distance-transform value above the saddle "
                         "(lowest ridge point) to its nearest taller neighbor. Distinguishes "
                         "a real second cell (deep saddle) from a noise bump on the shoulder "
                         "of one cell (shallow saddle), so tightly-clustered real cells whose "
                         "centers are closer together than --watershed_min_peak_dist can "
                         "still be split correctly -- which plain distance-based filtering "
                         "can never do (mechanically impossible once a cluster's own extent "
                         "is smaller than min_peak_dist). Default 0.25, confirmed on real "
                         "footage together with --watershed_fg_thresh 0.7 (also the default) "
                         "as a good split setting for this assay.")
    ap.add_argument("--watershed_use_intensity", action="store_true",
                    help="Find split seeds from raw grayscale brightness within each blob "
                         "instead of the mask's distance transform. Use when a whole cluster "
                         "of cells has fused into one undifferentiated mask blob with no "
                         "internal shape structure at all (confirmed on real footage: no "
                         "--mask_thresh recovers a gap once this happens) -- distance-based "
                         "splitting, prominence or not, cannot find a boundary that isn't "
                         "geometrically present in the mask, but the original brightness "
                         "peaks of each cell can still be distinct even when the mask isn't. "
                         "Combine with --watershed_prominence_frac. Off by default.")
    ap.add_argument("--watershed_est_cell_area", type=float, default=None,
                    help="Typical area (px^2) of ONE cell. Last-resort fallback for blobs "
                         "where even --watershed_use_intensity finds no distinguishable peak "
                         "at all (confirmed on real footage: a big saturated multi-cell "
                         "cluster with only 1 dot despite clearly containing several visible "
                         "cells). When set, any blob whose area implies more cells than were "
                         "actually found (round(blob_area / watershed_est_cell_area) > seeds "
                         "found) is force-split into that many pieces by partitioning the "
                         "blob's own pixels with k-means -- i.e. dividing the area into "
                         "one-cell-sized chunks -- rather than leaving it as a single "
                         "detection. Positions from this path are approximate (a geometric "
                         "guess, not a real peak), so only use it to fix the count on boxes "
                         "you've confirmed are merging multiple visible cells; get the area "
                         "value from print_blob_size_diagnostics' median blob size, or measure "
                         "one isolated cell directly. Default None = disabled.")

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
                         "Default: just --watershed_min_peak_dist / its auto default. Ignored "
                         "if --sweep_prominence_frac is given.")
    ap.add_argument("--sweep_prominence_frac", type=str, default=None,
                    help="Comma-separated watershed_prominence_frac values to compare side by "
                         "side in the --preview_frame_s image (e.g. '0.1,0.15,0.2,0.25'). "
                         "Replaces --sweep_min_peak_dist as the grid's row axis when given; "
                         "--watershed_min_peak_dist is held fixed as the candidate-peak "
                         "spacing (small, e.g. 4-5) for every tile.")

    ap.add_argument("--min_track_frames_for_speed", type=int, default=3)
    ap.add_argument("--allow_single_frame_count",   action="store_true")
    ap.add_argument("--min_seen_count", type=int, default=1,
                    help="Minimum number of frames a detection must be matched across "
                         "before it counts as a cell at all (in ANY bucket -- speedbin, "
                         "short_track, or streak_only), not just before it gets a speed "
                         "bin. Default 1 counts any detection, including one-frame blips. "
                         "Raise to e.g. 2 to filter out one-frame noise (compression/sensor "
                         "artifacts that happen to have cell-like size/shape) at the cost of "
                         "also dropping any real cell only visible for a single frame. "
                         "Overridden by --allow_single_frame_count for passed_line cells "
                         "specifically, since that flag's whole purpose is to permit instant "
                         "single-frame counts.")

    ap.add_argument("--enable_streak",       action="store_true")
    ap.add_argument("--streak_ar",           type=float, default=2.2)
    ap.add_argument("--streak_min_len",      type=float, default=4)
    ap.add_argument("--streak_min_area",     type=float, default=6)
    ap.add_argument("--streak_merge_dist",   type=float, default=25)
    ap.add_argument("--streak_merge_window", type=int,   default=2)

    ap.add_argument("--dedup_window", type=float, default=0.15,
                    help="Time window (s) for treating a re-exit near the same spot as the "
                         "same cell, not a new one. Only applies to tracks whose speed is above "
                         "--dedup_slow_speed_thresh -- see --dedup_window_slow for the slow case, "
                         "which this default is deliberately too short for.")
    ap.add_argument("--dedup_window_slow", type=float, default=150.0,
                    help="Like --dedup_window, but used instead whenever either the new exit or "
                         "the earlier one being compared against has speed <= "
                         "--dedup_slow_speed_thresh. Confirmed necessary on real footage: a "
                         "slow/adhesion-interacting cell can flicker in and out of detection "
                         "(a known MOG2 weakness -- near-static objects can get intermittently "
                         "absorbed into the background model) over tens of seconds, each "
                         "reappearance otherwise counted as a brand new cell. Confirmed directly: "
                         "the same physical cell counted 3 separate times, 68-90s apart, right at "
                         "the exit boundary, each a 1-frame track with speed=0. Default 150s "
                         "covers the widest gap seen so far; a fixed window can't rule out an "
                         "even longer gap, or (the trade-off in the other direction) wrongly "
                         "merging two genuinely different slow cells that happen to exit near the "
                         "same spot more than --dedup_dist apart in time but within this window -- "
                         "kept separate from --dedup_window specifically so fast cells (which "
                         "cross the exit zone in under a second and are in no danger of a real "
                         "second cell coincidentally landing on the same spot within 150s) aren't "
                         "exposed to that trade-off.")
    ap.add_argument("--dedup_slow_speed_thresh", type=float, default=5.0,
                    help="Speed threshold (same units as --m_per_px implies, else px/s) below "
                         "which a track uses --dedup_window_slow instead of --dedup_window for "
                         "duplicate-exit checks. Default 5 px/s. Note a track too short to trust "
                         "for speed (see --min_track_frames_for_speed) computes speed as 0 by "
                         "construction (no measurable displacement over 1 frame) -- this is "
                         "deliberate, not a bug: exactly these low-confidence short tracks are "
                         "what the confirmed triple-counting case looked like, so treating them "
                         "as 'possibly slow' for dedup purposes is the intended, conservative "
                         "behavior, not an artifact to work around.")
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
    ap.add_argument("--verify_crossings_out", default=None,
                    help="Save a contact sheet of small thumbnails, one per counted cell, "
                         "cropped from the real frame at the moment it was counted. Lets "
                         "you sanity-check that each count is a real cell by looking at one "
                         "image instead of trying to manually watch and tally crossings in "
                         "the full video. Only checks precision (false positives), not "
                         "recall (cells that were missed won't appear here).")
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
    ap.add_argument("--label_detections", action="store_true",
                    help="With --draw_detections, also number each detection box "
                         "(0, 1, 2, ... in det_boxes order for that frame) so a specific "
                         "box can be pointed at by number in feedback/review instead of "
                         "by pixel coordinates.")
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

    if args.dead_cell_relative_thresh is not None:
        ref, n_samples, sampled_s = calibrate_dead_cell_reference(
            args.video, warmup_start, start_frame, fps, args)
        if ref is None:
            print(f"WARNING: --dead_cell_relative_thresh set but no detections found in "
                  f"{sampled_s:.0f}s of calibration sampling -- dead-cell tagging disabled "
                  f"for this run.")
        else:
            args.dead_cell_max_sharpness = args.dead_cell_relative_thresh * ref
            if n_samples < 200:
                print(f"WARNING: dead-cell calibration only found {n_samples} samples in "
                      f"{sampled_s:.0f}s (wanted >=200) -- this video's start is low-activity, "
                      f"so the reference below may be noisy. Consider passing a later --start_s "
                      f"if this run's counts look off.")
            print(f"Auto-calibrated dead-cell threshold for this video: reference sharpness "
                  f"(p75 of {n_samples} real detections over {sampled_s:.0f}s) = {ref:.1f}, "
                  f"--dead_cell_relative_thresh {args.dead_cell_relative_thresh:g} -> "
                  f"effective cutoff = {args.dead_cell_max_sharpness:.1f}")

    varT_schedule = None  # list of (frame_idx, value), sorted, consumed in main loop
    if args.varThreshold_schedule:
        parsed = []
        for part in args.varThreshold_schedule.split(","):
            t_str, v_str = part.split(":")
            parsed.append((int(float(t_str) * fps), float(v_str)))
        varT_schedule = sorted(parsed)
        # apply every entry at/before start_frame now (skip straight to the right value
        # instead of replaying the schedule from t=0 if --start_s begins partway through it)
        while varT_schedule and varT_schedule[0][0] <= start_frame:
            args.mog2_varThreshold = varT_schedule.pop(0)[1]
        print(f"Using fixed --varThreshold_schedule: starting at "
              f"{args.mog2_varThreshold:g}, {len(varT_schedule)} more switch(es) ahead.")
    elif args.auto_mog2_varThreshold:
        best_varT, stats = calibrate_mog2_varThreshold(
            args.video, warmup_start, start_frame, fps, args)
        args.mog2_varThreshold = best_varT
        print("Auto-calibrated --mog2_varThreshold for this video/run:")
        for varT in sorted(stats, reverse=True):
            n, oversized_frac, sliver_frac, growth = stats[varT]
            flag = " <- selected" if varT == best_varT else ""
            print(f"  varThreshold={varT:g}: n_detections={n}, oversized_frac={oversized_frac:.2f}, "
                  f"sliver_frac={sliver_frac:.2f}, growth_vs_prev={growth:.2f}{flag}")

    recal_every_frames = (int(args.recalibrate_varThreshold_every_s * fps)
                           if (args.auto_mog2_varThreshold and not varT_schedule
                               and args.recalibrate_varThreshold_every_s > 0)
                           else None)
    next_recal_frame = start_frame + recal_every_frames if recal_every_frames else None

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
    dead_cell_counts   = defaultdict(int)
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

    def is_duplicate_exit(t, x, y, v):
        # Drop stale entries by scanning the whole buffer rather than just the front:
        # a lingering cell refreshes its entry's timestamp below, so the buffer is not
        # guaranteed sorted by time and a front-only pop would leave old entries stuck.
        # Pruning uses the longer of the two windows since either side of a pairwise
        # comparison below might turn out to need it -- the per-pair effective window
        # is decided inside the loop, not here.
        max_window = max(args.dedup_window, args.dedup_window_slow)
        kept = [(t0, x0, y0, v0) for (t0, x0, y0, v0) in recent_exits if t - t0 <= max_window]
        for i, (t0, x0, y0, v0) in enumerate(kept):
            # Either side being slow (or speed-unknown, which computes as 0 -- see
            # --dedup_slow_speed_thresh) means this pair could plausibly be the same
            # intermittently-detected cell, so use the generous window; only use the
            # short one when both sides are confidently fast.
            is_slow_pair = v <= args.dedup_slow_speed_thresh or v0 <= args.dedup_slow_speed_thresh
            eff_window = args.dedup_window_slow if is_slow_pair else args.dedup_window
            if t - t0 <= eff_window and math.hypot(x - x0, y - y0) <= args.dedup_dist:
                # Refresh instead of leaving the original timestamp: a cell that just
                # sits in the exit margin re-triggers crossed_exit() every frame (its
                # track gets deleted and immediately recreated), and without refreshing,
                # this entry would expire after exactly one dedup_window and let the
                # same still-lingering cell be counted again as a new exit. Keep the
                # slower of the two speeds seen so far: once a sighting of this spot
                # was slow, later fast-looking re-detections (e.g. a 1-frame blip with
                # speed 0) shouldn't accidentally graduate the entry back to the short
                # window.
                kept[i] = (t, x0, y0, min(v0, v))
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
        v = float(compute_speed(st))
        if is_duplicate_exit(t_abs, x_e, y_e, v):
            return False
        allow_instant = args.allow_single_frame_count and reason == "passed_line"
        if not allow_instant and st["seen_count"] < args.min_seen_count:
            # A detection only ever matched this few times was never confirmed as a
            # persistent, real object -- could be a single-frame noise blip (compression
            # artifact, sensor noise) that happens to have cell-like size/shape. Drop it
            # entirely rather than counting it in short_track/streak_only_short, and
            # don't record it in recent_exits either so it can't block a genuine later
            # detection at the same spot from being counted via dedup.
            return False
        recent_exits.append((t_abs, x_e, y_e, v))
        if len(recent_exits) > args.dedup_keep:
            recent_exits.popleft()
        tb  = int(t_rel // args.bin_seconds)
        thr = 1 if allow_instant else args.min_track_frames_for_speed
        if st["seen_count"] >= thr:
            sb = bin_index(v, edges)
            if sb is not None:
                counts[(tb, sb)] += 1
            counted_as = "speedbin"
        else:
            bucket = streak_only_counts if st["is_streak"] else short_track_counts
            bucket[tb] += 1
            counted_as = "streak_only_short" if st["is_streak"] else "short_track"
        # Majority vote across the track's own lifetime, not just its last frame --
        # a cell that measured as blurry (see --dead_cell_max_sharpness) in most
        # of the frames it was seen in is classified as a dead cell here, tagged
        # alongside (not instead of) its speed-bin/streak/short bucket above,
        # since live/dead is a separate dimension from how it was counted.
        is_dead_cell = st["dead_count"] * 2 >= st["seen_count"]
        if is_dead_cell:
            dead_cell_counts[tb] += 1
        # Always record a per_rows entry regardless of which bucket it landed in --
        # short_track/streak cells are the ones most likely to be noise (only tracked
        # a frame or two), so they're exactly the ones worth being able to verify,
        # not ones to silently omit from per_object_csv/--verify_crossings_out.
        if args.per_object_csv or args.verify_crossings_out:
            per_rows.append(dict(
                track_id=tid, final_reason=reason, counted_as=counted_as,
                seen_count=st["seen_count"], is_streak=int(st["is_streak"]),
                is_dead_cell=int(is_dead_cell),
                speed=v, speed_unit="m/s" if args.m_per_px else "px/s",
                time_exit_s_abs=t_abs, time_exit_s_rel=t_rel,
                x_exit=x_e, y_exit=y_e,
                x_start=float(st["start_x"]), y_start=float(st["start_y"])))
        return True

    while True:
        if end_frame_excl is not None and abs_frame >= end_frame_excl:
            break
        # --frame_stride cheaply skips decode+MOG2+tracking entirely on frames that
        # aren't a multiple of the stride away from warmup_start, via cap.grab()
        # (discards the frame without decoding it -- far cheaper than cap.read(),
        # which is the actual dominant cost being cut here). abs_frame still counts
        # every real frame at the video's true fps, so all time/speed math (which
        # is frame-index-based, not loop-iteration-based) stays correct unchanged;
        # only the effective time gap between *processed* frames grows, which is
        # why --max_dist likely needs increasing roughly proportionally to
        # --frame_stride (a cell now moves stride-times further between the frames
        # the tracker actually sees).
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
        last_processed = cur_frame

        while varT_schedule and cur_frame >= varT_schedule[0][0]:
            args.mog2_varThreshold = varT_schedule.pop(0)[1]
            print(f"t={cur_frame/fps:.0f}s: --varThreshold_schedule switching "
                  f"--mog2_varThreshold to {args.mog2_varThreshold:g}")
            backsub.setVarThreshold(args.mog2_varThreshold)

        if next_recal_frame is not None and cur_frame >= next_recal_frame:
            # Own throwaway VideoCapture/MOG2 instance (same reasoning as the initial
            # calibration -- MOG2 is stateful, can't share the live one), warmed up
            # from a bit before cur_frame rather than a full fresh warmup_s: this is a
            # mid-run recalibration checking how much the video's own characteristics
            # have drifted since the last one, not a cold start.
            recal_warmup_start = max(0, cur_frame - int(15 * fps))
            new_varT, recal_stats = calibrate_mog2_varThreshold(
                args.video, recal_warmup_start, cur_frame, fps, args)
            if new_varT != args.mog2_varThreshold:
                print(f"t={cur_frame/fps:.0f}s: re-calibrated --mog2_varThreshold "
                      f"{args.mog2_varThreshold:g} -> {new_varT:g}")
                args.mog2_varThreshold = new_varT
                backsub.setVarThreshold(new_varT)
            next_recal_frame = cur_frame + recal_every_frames

        if args.preview_frame_s is not None and cur_frame % 200 == 0:
            print(f"  ...scanned frame {cur_frame} (t={cur_frame/fps:.1f}s), "
                  f"target {start_s + args.preview_frame_s:.1f}s", flush=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.gauss_ksize > 0:
            k = args.gauss_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        fg = backsub.apply(gray, learningRate=float(args.learning_rate))
        _, th_mog2 = cv2.threshold(fg, args.mask_thresh, 255, cv2.THRESH_BINARY)

        # Raw frame-to-frame diff, kept separate from --use_framediff's mask-combination
        # role: used by --min_local_motion to reject detections with no current motion.
        # A MOG2 "ghost" -- a real region flagged as foreground because something (often
        # debris) was recently there, even though it has since moved on -- has ordinary
        # background-level intensity by the time it's detected, so --min_mean_intensity
        # can't catch it; only checking actual frame-to-frame change can.
        raw_diff = cv2.absdiff(gray, prev_gray) if prev_gray is not None else None

        th_diff = None
        if args.use_framediff and raw_diff is not None:
            _, th_diff = cv2.threshold(raw_diff, args.diff_thresh, 255, cv2.THRESH_BINARY)
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
            save_preview(th, frame, gray, args, raw_diff)
            cap.release()
            if vw: vw.release()
            return

        contours, detections, det_boxes = detect_cells(th, frame, gray, args, raw_diff=raw_diff)

        for tid in tracks:
            tracks[tid]["updated"] = False

        pairs, unmatched = hungarian_match(tracks, detections, args.max_dist)

        for tid, j in pairs:
            cx, cy, is_streak, is_dead = detections[j]
            st = tracks[tid]
            st["cx"], st["cy"] = cx, cy
            st["last_seen_frame"] = cur_frame
            st["seen_count"]  += 1
            st["missed_count"] = 0
            st["updated"]      = True
            st["is_streak"]    = st["is_streak"] or is_streak
            st["dead_count"]  += 1 if is_dead else 0
            st["end_x"], st["end_y"] = cx, cy

        for j in unmatched:
            cx, cy, is_streak, is_dead = detections[j]
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
                dead_count=1 if is_dead else 0,
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
                    # Only count a lost track if it was actually near the exit boundary
                    # when lost (genuinely mid-crossing, e.g. fragmented right at the
                    # edge) -- same reasoning as end_of_range. A track lost anywhere
                    # else in the frame (occlusion, a brief detection gap, MOG2 losing
                    # it mid-frame) never demonstrated it was exiting and should just be
                    # dropped, not counted as a cell that passed the FOV. Confirmed via
                    # --verify_crossings_out: a "missing" cell counted near the bottom
                    # of a "top"-exit frame, nowhere near the boundary.
                    if near_exit(st["cx"], st["cy"], args.end_of_range_margin_px):
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
                for i, (x,y,w,h,is_s,is_dead) in enumerate(det_boxes):
                    col = (0,0,255) if is_dead else ((0,255,255) if is_s else (255,0,255))
                    cv2.rectangle(left,(x,y),(x+w,y+h),col,1)
                    if args.label_detections:
                        cv2.putText(left,str(i),(x,y-3),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)

            for tid,st in tracks.items():
                dead = st["dead_count"] * 2 >= st["seen_count"]
                col = (0,0,255) if dead else ((0,255,0) if not st["is_streak"] else (0,255,255))
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
            + ["streak_only_short","short_track","total_counted","dead_cell_count"])

    rows = []
    for tb in range(max_tb+1):
        row = [start_s+tb*args.bin_seconds, start_s+(tb+1)*args.bin_seconds]
        tot = 0
        for sb in range(len(edges)-1):
            c=counts.get((tb,sb),0); row.append(c); tot+=c
        c=counts.get((tb,len(edges)-1),0); row.append(c); tot+=c
        cs=streak_only_counts.get(tb,0); ch=short_track_counts.get(tb,0)
        # dead_cell_count is a tag on cells already included in the counts above
        # (a subset, not an additional population) -- do not add it into totals.
        dc=dead_cell_counts.get(tb,0)
        row+=[cs,ch,tot+cs+ch,dc]; rows.append(row)

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
    if args.verify_crossings_out:
        save_crossing_contact_sheet(args.video, per_rows, fps, args.verify_crossings_out)


if __name__ == "__main__":
    main()
