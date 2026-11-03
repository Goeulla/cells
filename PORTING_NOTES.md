# Porting notes (Python 2 / OpenCV 2.4 -> Python 3 / OpenCV 4, headless)

Original source: https://github.com/telescope7/TrafficFlowAnalysis
(Matthew Thomas, `traffic_analyzer.py`, master branch, commit `012a457`).

This copy has been ported to run on Python 3 with modern OpenCV (tested on
opencv-python-headless 5.0), and further adapted to run headless in Google
Colab and other display-less environments (this repo's dev sandbox
included). Behavior and the tracking algorithm are otherwise unchanged from
the original.

## Changes made (Python 3 / OpenCV 4 port)

1. **Removed `import cv2.cv as cv`** (the old OpenCV 1.x-style API module
   was removed in OpenCV 3+). Replaced the trackbar/window calls
   (`cv.NamedWindow`, `cv.CreateTrackbar`, `cv.SetTrackbarPos`) with their
   `cv2` equivalents (`cv2.namedWindow`, `cv2.createTrackbar`,
   `cv2.setTrackbarPos`).
2. **`print` statements -> `print()` function calls.**
3. **`dict.itervalues()` -> `dict.values()`** (Python 2 only had the
   iterator form).
4. **`cv2.cv.CV_FOURCC(...)` -> `cv2.VideoWriter_fourcc(...)`.**
5. **Hard-coded property IDs (`capture.get(3)`, `capture.get(4)`) ->
   named constants** (`cv2.CAP_PROP_FRAME_WIDTH` / `_HEIGHT`) for clarity;
   same underlying values.
6. **Fixed a real bug surfaced by the Python 3 port**: `MovingObject.measure()`
   and `MovingObject.__str__()` did
   `sorted(self.frames, key=self.frames.get)` — sorting the frame-number
   dict keys using the corresponding `Point` object as the sort key. `Point`
   has no `__lt__`, so this raises `TypeError` in Python 3 (Python 2 silently
   fell back to an arbitrary, non-meaningful ordering by type/id). Since
   `self.frames` is keyed by frame number and the code only ever needs
   `min()`/`max()`/`len()` of those keys, this didn't actually depend on a
   meaningful sort order — it's fixed to `sorted(self.frames)` (sorting the
   frame numbers directly), which is both correct and matches the original
   intent.
7. **Minor robustness fix**: the main capture loop now breaks cleanly if
   `capture.read()` fails or returns no frame, instead of passing `None`
   into `cv2.cvtColor`. The initial capture-then-reset also checks the
   first `capture.read()`, and `cv2.VideoCapture.isOpened()` is checked
   right after opening the movie, so a bad/missing path fails with a clear
   log message instead of a `cv2.error` deep inside the frame loop.
8. Replaced a dead `next` statement (a no-op in the original — it just
   evaluated the `next` builtin and discarded it) in the ESC-key handler of
   the interactive `-v/--visualize` mode with `continue`, so pressing ESC
   during playback actually skips the rest of that iteration as intended.
   This only affects interactive visualization; it does not affect batch
   (non-`-v`) analysis, which is the normal way this tool is run for
   RTD/cell-tracking data.

## Changes made (headless / Colab compatibility)

The interactive `-v/--visualize` mode was built entirely on OpenCV's
HighGUI (`cv2.namedWindow`, `cv2.createTrackbar`, `cv2.imshow`,
`cv2.waitKey`). That backend is simply absent in most environments this
tool now needs to run in:

* `opencv-python-headless` — the wheel that reliably installs in Colab, CI,
  and containers without a display, and the one this fork's
  `requirements-traffic_analyzer.txt` now pins instead of plain
  `opencv-python` — has **no HighGUI support at all**. Any HighGUI call,
  including `cv2.destroyAllWindows()`, raises `cv2.error`.
* Even a regular `opencv-python` build fails the same way with no `$DISPLAY`
  (e.g. this repo's own dev sandbox).

Unpatched, this meant `cv2.destroyAllWindows()` at the very end of `main()`
crashed **every** run — including plain batch/CSV analysis with no `-v` at
all — as soon as the movie finished, on any headless install. Fixed as
follows:

9. **Added `gui_available()`**, which probes `cv2.namedWindow` in a
   `try/except cv2.error` once at startup, instead of assuming a GUI
   backend exists.
10. **`cv2.destroyAllWindows()` is now only called when a GUI window was
    actually opened** (`has_gui`), fixing the crash-on-exit described above.
11. **`-v/--visualize` degrades instead of crashing** when no GUI backend is
    available:
    * In Google Colab (detected via `'google.colab' in sys.modules`), it
      falls back to showing every Nth annotated frame inline via
      `google.colab.patches.cv2_imshow` (`--colab_preview_stride`, default
      30 — showing every single frame would flood notebook output and be
      very slow). There are no live trackbars in this mode; pass `-b`/`-t`
      explicitly.
    * Anywhere else with no display, `-v` is disabled with a logged warning
      and the run proceeds as a normal headless batch analysis. Use `-o` to
      still get an annotated output video to inspect afterward.
12. Trackbar creation (`cv2.namedWindow`/`createTrackbar`/`setTrackbarPos`)
    is now also gated on `has_gui`, so it isn't attempted at all when there's
    no HighGUI backend.

None of this changes the detection/tracking algorithm or the CSV output
format — only how (and whether) frames get displayed interactively.

## Verified

Ran against the repo's own sample video (`videos/traffic.avi`, 1920x540,
1245 frames @ 24fps) with parameters resembling the RTD paper's method
(`-b 13 -t 5 -a 0.0005 -s`, i.e. blur 13, threshold 5, background-subtraction
accumulator weight 0.0005): the script runs without errors and streams valid
CSV rows (num_frames, first_x, first_y, last_x, last_y, first_frame,
last_frame, avg_radius, avg_width, avg_height, avg_area) to stdout as
objects finish being tracked.

Headless behavior verified with a synthetic test video and
`opencv-python-headless`: batch mode (no `-v`), `-v` with no display
(warns and disables, doesn't crash), and `-v` under a simulated
`google.colab` import (falls back to inline `cv2_imshow` preview at the
configured stride) — all complete with exit code 0 and no `cv2.error`.

**Performance note**: the tracking/stitching logic is O(n^2)-ish per frame
over currently-tracked objects, inherited from the original design — on the
1245-frame, many-object sample traffic video it took several minutes to
process. Cell-tracking RTD video will likely have different object counts/
density, so actual runtime on your data may differ substantially.

## Method, per the RTD paper this fork is used for

> Videos from residence time distribution experiments were analyzed by
> modifying the OpenCV-based Traffic Flow Analyzer. Briefly, the background
> was subtracted using a mask weight of 0.0005, resulting in an approximate
> 80 s moving average. A blurring factor of 13 and a thresholding factor of
> 5 were used. The contour detection algorithm defines cells as >5 μm in
> diameter that are stored in the Moving Object database and compared to
> other previously tracked objects and mapped to the appropriate moving
> instance based on forward movement (using either a look-ahead window or an
> overlapping object boundary analysis). Once the object is no longer
> tracked or exits the FOV, the object data is read to file. Data were
> included in the cell velocity analysis if the tracking distance was >400
> pixels, the cell diameter between 14 and 60 μm and the cell within 50 μm
> of the chamber bottom (as calculated based on its tracked velocity).

## Usage

```
python3 traffic_analyzer.py -m <video> -c <diameter_px> -b <blur> \
    -t <threshold> -s -a <accumulator_weight> > output.csv
```

Key options relevant to reproducing paper-style analysis:

* `-s` enable background subtraction
* `-a` accumulator weight (paper: 0.0005, ~80s moving average)
* `-b` Gaussian blur kernel size (paper: 13)
* `-t` threshold (paper: 5)
* `-c` minimum object diameter **in pixels** (the paper's 5 um cutoff for
  contour detection, and the 14-60 um / >400 px tracking-distance / within
  50 um of chamber bottom filters, are downstream physical-unit filters —
  you'll need your video's um-per-pixel calibration to convert those to
  pixel values, and to convert this tool's raw pixel output back to
  physical units and velocities afterward. This script does not do that
  conversion itself.)

Add `-v` to visualize tracking (interactive window where available; falls
back to periodic inline preview in Colab, or is disabled with a warning if
there is no display at all), `-o` to write an annotated output video, `-d`
for debug logging.

See `TrafficFlowAnalyzer_Colab.ipynb` for a ready-to-run Google Colab
notebook that installs dependencies, uploads a video, runs this script with
the paper's parameters, and plots the resulting tracks.
