"""
count_static_cells.py -- count cells in still phase-contrast images (screenshots or
single frames), no video or background model needed.

Standalone: does not import or change the video counting/tracking scripts.

How it works
  1. Crop to the image area: rows/columns that are much darker than the rest (video
     player bars, black borders) are trimmed automatically (or give --crop).
  2. Each cell is a bright halo ring roughly one cell diameter across. The image is
     resampled so a cell is ~7 px wide, and a difference-of-Gaussians (DoG) tuned to
     that size turns every halo into one positive peak, also when cells touch.
     Peaks above --threshold grey levels are counted as cells.
  3. Strong negative peaks (dark cells with little or no halo: dead, debris or out of
     focus) are counted separately as "dark spots", not as cells. A bright peak right
     beside a dark spot is that dark cell's halo and is not counted as a cell.

The cell diameter in pixels is estimated from the image (autocorrelation of the
high-passed image) unless you give --cell_diameter_px. It depends on how much the
image was enlarged: ~7 px in a 330 px-wide thumbnail, ~37-40 px in an 1800 px-wide
screenshot. Count single frames, not a montage of several frames.

Accuracy, checked by eye on two 1800 px screenshots: most cells are found; touching
pairs are sometimes counted as one, and faint halo rings or debris are sometimes
counted as a cell, so expect roughly +-10-15%. Always look at the overlay.

Usage
  python3 count_static_cells.py image1.png [image2.webp ...] --out_dir counts/
  python3 count_static_cells.py frame.png --cell_diameter_px 38 --threshold 8

Outputs, per image: <name>_count.png (green = cell, magenta = dark spot) and
<name>_cells.csv (x, y, kind); plus counts_summary.csv for all images. Default
--out_dir is a "static_counts" folder next to the first image.
"""
import argparse
import csv
import os

import cv2
import numpy as np
from scipy import ndimage as ndi


def auto_crop(gray, frac=0.8):
    """Longest run of rows and of columns whose mean brightness is >= frac x the
    median row/column mean -- trims dark player bars and black borders."""
    def run(prof):
        ok = prof >= frac * np.median(prof)
        best, s = (0, len(prof)), None
        best_len = 0
        for i, v in enumerate(np.append(ok, False)):
            if v and s is None:
                s = i
            elif not v and s is not None:
                if i - s > best_len:
                    best, best_len = (s, i), i - s
                s = None
        return best
    y0, y1 = run(gray.mean(1))
    x0, x1 = run(gray.mean(0))
    return x0, y0, x1 - x0, y1 - y0


REF_D = 7.0   # cell diameter (px) the detector is tuned for; images are resampled to it
SIGMA = 1.5   # DoG inner sigma at REF_D


def dog(g, s1):
    return cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), 2.5 * s1)


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


def count_image(path, args):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"cannot read {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    bgr = im[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if args.crop:
        x0, y0, w, h = [int(v) for v in args.crop.split(",")]
    elif args.no_autocrop:
        x0, y0, w, h = 0, 0, gray.shape[1], gray.shape[0]
    else:
        x0, y0, w, h = auto_crop(gray)
    g = gray[y0:y0 + h, x0:x0 + w]

    d = args.cell_diameter_px or estimate_diameter(g)
    # Resample so a cell is REF_D px across, where the detector settings were checked
    # by eye; at high zoom a cell's own bright/dark sub-structure would otherwise
    # split it into several peaks.
    k = REF_D / d
    gs = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    r = dog(gs, SIGMA)
    md = max(1, int(round(args.min_dist_frac * REF_D)))
    edge = 2  # ignore peaks hugging the crop edge
    h, w = gs.shape

    mx = ndi.maximum_filter(r, size=2 * md + 1)
    cells = np.argwhere((r == mx) & (r > args.threshold))
    mn = ndi.minimum_filter(r, size=2 * md + 1)
    dark = np.argwhere((r == mn) & (r < -args.dark_threshold))
    inside = lambda p: p[(p[:, 0] >= edge) & (p[:, 0] < h - edge) & (p[:, 1] >= edge) & (p[:, 1] < w - edge)]
    cells, dark = inside(cells), inside(dark)
    if len(cells) and len(dark):
        dist = np.sqrt(((dark[:, None, :] - cells[None, :, :]) ** 2).sum(-1))
        # a dark peak right at a bright peak is that cell's own dark centre -> drop it
        dark = dark[dist.min(1) > 0.35 * REF_D]
        dist = np.sqrt(((dark[:, None, :] - cells[None, :, :]) ** 2).sum(-1))
        # a bright peak just beside a dark spot is the dark cell's halo, not a second
        # cell -> drop the bright peak (a touching neighbour sits ~1 diameter away)
        if len(dark):
            cells = cells[dist.min(0) > 0.7 * REF_D]
    cells, dark = cells / k, dark / k          # back to original-image pixels
    h, w = g.shape

    stem = os.path.splitext(os.path.basename(path))[0]
    vis = bgr.copy()
    cv2.rectangle(vis, (x0, y0), (x0 + w - 1, y0 + h - 1), (0, 200, 255), 1)
    rad = max(2, int(round(0.45 * d)))
    lw = 1 if d < 15 else 2
    for py, px in cells:
        cv2.circle(vis, (int(px + x0), int(py + y0)), rad, (0, 255, 0), lw)
    for py, px in dark:
        cv2.circle(vis, (int(px + x0), int(py + y0)), rad, (255, 0, 255), lw)
    label = f"cells: {len(cells)}   dark spots: {len(dark)}"
    fs = max(0.6, w / 1500)
    cv2.putText(vis, label, (x0 + 8, y0 + int(30 * fs)), 0, fs, (0, 0, 0), int(5 * fs) + 1)
    cv2.putText(vis, label, (x0 + 8, y0 + int(30 * fs)), 0, fs, (0, 255, 255), max(1, int(2 * fs)))
    cv2.imwrite(os.path.join(args.out_dir, f"{stem}_count.png"), vis)
    with open(os.path.join(args.out_dir, f"{stem}_cells.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["x", "y", "kind"])
        wr.writerows([(int(px + x0), int(py + y0), "cell") for py, px in cells])
        wr.writerows([(int(px + x0), int(py + y0), "dark_spot") for py, px in dark])
    return dict(image=path, cells=len(cells), dark_spots=len(dark),
                cell_diameter_px=round(d, 1), crop=f"{x0},{y0},{w},{h}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out_dir", default=None,
                    help="Default: static_counts/ next to the first image.")
    ap.add_argument("--cell_diameter_px", type=float, default=None,
                    help="Cell (halo) diameter in pixels; estimated from the image if omitted.")
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="Minimum DoG peak height in grey levels for a cell. Lower = more "
                         "faint cells but also faint halo rings and specks; higher = fewer "
                         "false cells but dim cells are missed.")
    ap.add_argument("--min_dist_frac", type=float, default=0.3,
                    help="Minimum distance between two counted cells, as a fraction of the "
                         "diameter. Larger merges touching cells; smaller splits single cells.")
    ap.add_argument("--dark_threshold", type=float, default=10.0,
                    help="Minimum negative DoG peak depth (grey levels) for a dark spot.")
    ap.add_argument("--crop", default=None, help="x,y,w,h region to count (default: automatic).")
    ap.add_argument("--no_autocrop", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.images[0])), "static_counts")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [count_image(p, args) for p in args.images]
    with open(os.path.join(args.out_dir, "counts_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"{r['image']}: {r['cells']} cells, {r['dark_spots']} dark spots "
              f"(cell diameter {r['cell_diameter_px']} px, counted region x,y,w,h = {r['crop']})")
    print(f"Overlays and per-cell CSVs in {args.out_dir}/")


if __name__ == "__main__":
    main()
