Traffic Flow Analyzer is a program designed to detect and measure uni-directional objects in videos.

The software works by loading the specified video, applying a background subtraction if specified, blurring the frame by a specified amount, thresholding the frame by a specified amount, running a contour detection algorithm
and selecting objects that surpass the minimum object diameter requirement.  These objects will be added to the Moving Object database if not present.  Otherwise, the object is compared to other previously tracked objects
and mapped to the appropriate moving instance based on forward movement (using either a look ahead window or an overlapping object boundry analysis).  Once the object is no longer tracked or exits the field of view, the 
statistics and measurements of the tracked object are reported.  Each tracked object will be recorded on STDOUT.  The objects measurements include:

* Number of frames tracked
* First X coordinate 
* First Y coordinate
* Last X coordinate
* Last Y coordinate
* First frame tracked
* Last frame tracked
* Average radius of enclosing circle
* Average width of enclosing contour
* Average heigh of enclosing contour
* Average area of enclosing contour

This copy (`traffic_analyzer.py`) is a Python 3 / modern OpenCV port of the
original tool, further adapted to run headless — see `PORTING_NOTES.md` for
what changed and why.

# How to use

## Locally / any machine with Python 3

```
pip install -r requirements-traffic_analyzer.txt
python3 traffic_analyzer.py [OPTIONS]
```

'-d', '--debug', debug mode

'-m', '--movie', movie.avi

'-b', '--blur', blur setting

'-t', '--threshold', threshold

'-c', '--objectdiametermin', object minimum diameter

'-x', '--boxwindow', box window size

'-p', '--playbackspeed'  playback speed

'-v', '--visualize', visualize tracking (interactive window if a display is
available; otherwise falls back to periodic inline preview frames in
Google Colab, or is disabled with a warning if there's no display at all)

'-s', '--subtractbg', subtract background

'-a', '--accumulator', accumulator weight

'-o', '--output', output avi

'--colab_preview_stride', how often (in frames) to show an inline preview
under `-v` in Colab, since there's no live interactive window there
(default: 30)

Example:

    python3 traffic_analyzer.py -c 315 -p 35 -b 45 -t 30 -s -m videos/traffic.avi -v

Example reproducing the RTD paper's method (see `PORTING_NOTES.md`):

    python3 traffic_analyzer.py -m video.avi -c <px_for_5um> -b 13 -t 5 -s -a 0.0005 > tracks.csv

## Google Colab

Open `TrafficFlowAnalyzer_Colab.ipynb` in Colab (File -> Open notebook ->
GitHub, or upload it directly) and run the cells in order. It installs
`opencv-python-headless`/`numpy`, lets you upload or point at a video, runs
`traffic_analyzer.py` with the RTD paper's parameters, and loads/plots the
resulting CSV. `cv2.imshow`/interactive trackbars don't work in Colab
(there's no display), so the notebook runs the analysis in batch mode and
uses `-o` to produce an annotated video you can review afterward, plus the
optional inline-preview mode described above.

![Example Visualization of Tracked Objects](https://raw.github.com/telescope7/TrafficFlowAnalysis/master/exampleTracking.png)

# Contact

Traffic Flow Analyzer was developed by [Matthew Thomas](https://github.com/telescope7/) 

