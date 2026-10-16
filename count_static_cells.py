"""
count_static_cells.py -- count cells in still phase-contrast images (screenshots or
single frames), no video or background model needed.

Standalone: does not import or change the video counting/tracking scripts.

How it works
  1. Crop to the image area: rows/columns that are much darker than the rest (video
     player bars, black borders) are trimmed automatically (or give --crop).
  2. The image is resampled so a cell is ~7 px wide. Local contrast energy (RMS of
     the background-subtracted image over a cell-sized patch) gives one peak per
     cell whether the cell looks bright (halo ring) or dark (dark body), and also
     when cells touch. Peaks above the threshold are counted.
  3. Each cell is labelled "bright" or "dark" by whether its centre is brighter or
     darker than the background; both count as cells.

The cell diameter in pixels is estimated from the image (autocorrelation of the
high-passed image) unless you give --cell_diameter_px. It depends on how much the
image was enlarged: ~7 px in a 330 px-wide thumbnail, ~37-40 px in an 1800 px-wide
screenshot. For a slide or figure with several pictures, use --panels; give
--cell_diameter_px then, as small pictures are too small to estimate it reliably.

Accuracy: on in-focus images, within ~5-10% of hand counts of test patches (dense
and sparse). It does NOT work on out-of-focus images where each cell is a set of
large concentric rings -- every ring arc is counted, giving several times too many.
Always look at the overlay.

Usage
  python3 count_static_cells.py image1.png [image2.webp ...] --out_dir counts/
  python3 count_static_cells.py frame.png --cell_diameter_px 38 --threshold 10
  python3 count_static_cells.py slide.png --panels --cell_diameter_px 7   # several pictures

Outputs, per image: <name>_count.png (green = bright cell, magenta = dark cell) and
<name>_cells.csv (panel, x, y, kind); plus counts_summary.csv for all images. Default
--out_dir is a "static_counts" folder next to the first image.
"""
import argparse
import csv
import os

import cv2
import numpy as np
from scipy import ndimage as ndi


def auto_crop(gray, frac=0.8):
    """Longest run of rows and of columns that belong to the picture -- trims video
    player bars and black borders. A row/column counts as picture if it is not much
    darker than typical (mean >= frac x median) OR it has texture (std >= half the
    typical std), so a vignetted but cell-filled edge is kept."""
    def run(mean, std):
        ok = (mean >= frac * np.median(mean)) | (std >= 0.5 * np.median(std))
        best, s = (0, len(mean)), None
        best_len = 0
        for i, v in enumerate(np.append(ok, False)):
            if v and s is None:
                s = i
            elif not v and s is not None:
                if i - s > best_len:
                    best, best_len = (s, i), i - s
                s = None
        return best
    y0, y1 = run(gray.mean(1), gray.std(1))
    x0, x1 = run(gray.mean(0), gray.std(0))
    return x0, y0, x1 - x0, y1 - y0


REF_D = 7.0     # cell diameter (px) the detector is tuned for; images are resampled to it
HP_SIGMA = 4.0  # background removal scale at REF_D
NMS = 2         # minimum peak separation (px at REF_D); touching cells are ~REF_D apart


def estimate_diameter(g):
    """Cell diameter (px) = radius at which the radially averaged autocorrelation of
    the high-passed image first crosses zero. For an image of many similar blobs this
    is ~ one blob diameter (checked: ~7 px on 330 px-wide thumbnails, ~37 px on
    1800 px-wide screenshots of the same kind of frame)."""
    g = g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), max(g.shape) / 40)
    F = np.fft.fft2(hp)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.indices(ac.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    prof = np.bincount(r.ravel(), ac.ravel()) / np.bincount(r.ravel())
    below = np.nonzero(prof[1:] <= 0)[0]
    return float(below[0] + 1) if len(below) else REF_D


def find_panels(gray, min_frac=0.01):
    """Separate pictures on a slide/figure: large non-white regions, left-to-right
    within rows, top to bottom. Returns a list of (x, y, w, h), inset by 4 px."""
    m = (gray < 200).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    _, _, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = [tuple(int(v) for v in s[:4]) for s in st[1:] if s[4] >= min_frac * gray.size]
    # pictures are roughly camera-shaped; drop banners, diagrams and text blocks
    boxes = [b for b in boxes if 0.5 <= b[2] / b[3] <= 2.5]
    if not boxes:
        return []
    # group into rows by vertical centre, then left to right
    hmed = np.median([b[3] for b in boxes])
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    rows, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if b[1] + b[3] / 2 - (cur[0][1] + cur[0][3] / 2) > 0.5 * hmed:
            rows.append(cur)
            cur = []
        cur.append(b)
    rows.append(cur)
    boxes = [b for row in rows for b in sorted(row, key=lambda b: b[0])]
    return [(x + 4, y + 4, w - 8, h - 8) for x, y, w, h in boxes]


def detect(g, args):
    """Cells in grey image g -> (bright yx array, dark yx array, diameter px)."""
    d = args.cell_diameter_px or estimate_diameter(g)
    # Resample so a cell is REF_D px across, where the detector settings were checked
    # against hand counts.
    k = REF_D / d
    gs = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    # Sharp, zoomed-in sources keep each cell's bright/dark sub-structure after
    # resampling, which needs a bit more smoothing to stay one peak per cell.
    smooth = args.smooth or (1.4 if k < 0.5 else 1.0)
    # Local contrast energy: RMS of the high-passed image over a cell-sized patch. A
    # cell gives one energy peak whatever it looks like -- bright ring, dark body, or
    # both -- so a dark cell's halo or a bright cell's dark centre is not a 2nd object.
    hp = gs - cv2.GaussianBlur(gs, (0, 0), HP_SIGMA)
    e = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), smooth))
    mx = ndi.maximum_filter(e, size=2 * NMS + 1)
    pk = np.argwhere((e == mx) & (e > 1))
    hts = e[pk[:, 0], pk[:, 1]]
    # threshold: a fixed floor, raised in high-contrast images to a fraction of the
    # strongest peaks so background texture is not counted
    thr = max(args.threshold, args.rel_threshold * np.percentile(hts, 95)) if len(hts) else args.threshold
    pk = pk[hts > thr]
    hh, ww = gs.shape
    pk = pk[(pk[:, 0] >= 2) & (pk[:, 0] < hh - 2) & (pk[:, 1] >= 2) & (pk[:, 1] < ww - 2)]
    # appearance: centre darker than the local background -> "dark" cell
    centre = cv2.GaussianBlur(hp, (0, 0), 1.0)[pk[:, 0], pk[:, 1]]
    return pk[centre >= 0] / k, pk[centre < 0] / k, d


def count_image(path, args):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"cannot read {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    bgr = im[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if args.panels:
        regions = find_panels(gray)
        if not regions:
            raise SystemExit(f"{path}: no pictures found on a white background")
    elif args.crop:
        regions = [tuple(int(v) for v in args.crop.split(","))]
    elif args.no_autocrop:
        regions = [(0, 0, gray.shape[1], gray.shape[0])]
    else:
        regions = [auto_crop(gray)]

    stem = os.path.splitext(os.path.basename(path))[0]
    vis = bgr.copy()
    rows, points = [], []
    for i, (x0, y0, w, h) in enumerate(regions):
        cells, dark, d = detect(gray[y0:y0 + h, x0:x0 + w], args)
        name = f"panel {i + 1}" if args.panels else ""
        rad = max(2, int(round(0.45 * d)))
        lw = 1 if d < 15 else 2
        cv2.rectangle(vis, (x0, y0), (x0 + w - 1, y0 + h - 1), (0, 200, 255), 1)
        for py, px in cells:
            cv2.circle(vis, (int(px + x0), int(py + y0)), rad, (0, 255, 0), lw)
        for py, px in dark:
            cv2.circle(vis, (int(px + x0), int(py + y0)), rad, (255, 0, 255), lw)
        label = (f"{i + 1}: {len(cells) + len(dark)}" if args.panels else
                 f"cells: {len(cells) + len(dark)}  (bright {len(cells)}, dark {len(dark)})")
        fs = max(0.5, w / 1500 if not args.panels else w / 600)
        cv2.putText(vis, label, (x0 + 6, y0 + int(28 * fs)), 0, fs, (0, 0, 0), int(5 * fs) + 1)
        cv2.putText(vis, label, (x0 + 6, y0 + int(28 * fs)), 0, fs, (0, 255, 255), max(1, int(2 * fs)))
        points += [(name, int(px + x0), int(py + y0), "bright") for py, px in cells]
        points += [(name, int(px + x0), int(py + y0), "dark") for py, px in dark]
        rows.append(dict(image=path, panel=i + 1 if args.panels else "",
                         cells=len(cells) + len(dark), bright=len(cells), dark=len(dark),
                         cell_diameter_px=round(d, 1), region=f"{x0},{y0},{w},{h}"))
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_count.png"), vis)
    with open(os.path.join(args.out_dir, f"{stem}_cells.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["panel", "x", "y", "kind"])
        wr.writerows(points)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out_dir", default=None,
                    help="Default: static_counts/ next to the first image.")
    ap.add_argument("--cell_diameter_px", type=float, default=None,
                    help="Cell (halo) diameter in pixels; estimated from the image if omitted.")
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="Minimum local contrast (RMS grey levels) for a cell. Lower = more "
                         "faint cells but also more specks/halo arcs; higher = the reverse.")
    ap.add_argument("--rel_threshold", type=float, default=0.35,
                    help="Also require this fraction of the image's 95th-percentile peak "
                         "contrast, so texture in high-contrast images is not counted.")
    ap.add_argument("--smooth", type=float, default=None,
                    help="Contrast-energy smoothing (px at a 7 px cell). Default 1.4 for "
                         "zoomed-in screenshots (cells > 14 px), else 1.0. Raise it if single "
                         "cells get two circles; lower it if touching cells get only one.")
    ap.add_argument("--panels", action="store_true",
                    help="The image is a slide/figure with several pictures on a white "
                         "background: find each picture and count it separately.")
    ap.add_argument("--crop", default=None, help="x,y,w,h region to count (default: automatic).")
    ap.add_argument("--no_autocrop", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.images[0])), "static_counts")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [r for p in args.images for r in count_image(p, args)]
    with open(os.path.join(args.out_dir, "counts_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        where = f" panel {r['panel']}" if r["panel"] != "" else ""
        print(f"{r['image']}{where}: {r['cells']} cells ({r['bright']} bright, {r['dark']} dark) "
              f"(cell diameter {r['cell_diameter_px']} px, region x,y,w,h = {r['region']})")
    print(f"Overlays and per-cell CSVs in {args.out_dir}/")


if __name__ == "__main__":
    main()
