#!/usr/bin/env python3
"""
Traffic Flow Analyzer (RTD / cell-tracking fork)

Ported from the original Python 2 / OpenCV 2.4 tool by Matthew Thomas
(https://github.com/telescope7/TrafficFlowAnalysis) to Python 3 / OpenCV 4,
and made to run headless (Google Colab, this sandbox, CI, etc.) where no
GUI/display is available -- see PORTING_NOTES.md.

Detects and tracks moving objects (e.g. cells in residence-time-distribution
video) frame by frame via background subtraction, blurring, thresholding and
contour detection, then reports per-object statistics as CSV to STDOUT.
"""

import sys
import logging
import logging.handlers
import optparse

import cv2
import numpy as np
from random import randrange

import csv
from collections import defaultdict


class Point(object):

    def __init__(self, frame, contour, cx, cy, radius):
        self.frame = frame
        self.contour = contour
        self.cx = cx
        self.cy = cy
        self.radius = radius
        (self.width, self.height,
         self.area) = self.measure_width_height_area(contour, radius)

    def __repr__(self):
        return (
            "CX:" + str(self.cx) + "\tCY:" + str(self.cy) +
            "\tRadius:" + str(self.radius)
        )

    def measure_width_height_area(self, contour, radius):

        area = cv2.contourArea(contour)
        width = height = radius * 2

        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            width = min(ellipse[1])
            height = max(ellipse[1])

        return (width, height, area)


class MovingObject(object):

    def __init__(self):
        self.frames = {}
        self.color = (randrange(255), randrange(255), randrange(255))

    def __str__(self):
        x = ' '.join(str(self.frames[f])
                     for f in sorted(self.frames))
        return x

    def add_frame(self, frame, contour):
        self.last_frame = frame
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        self.frames[frame] = Point(frame, contour, cx, cy, radius)

    def match_overlap(self, contour):
        # first check if the contour directly overlaps with a previous
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        lf = self.frames[self.last_frame]

        if cx > lf.cx + 6:
            return False

        x1, y1, w1, h1 = cv2.boundingRect(contour)
        x2, y2, w2, h2 = cv2.boundingRect(lf.contour)

        if x2 < x1:
            x1, y1, w1, h1 = cv2.boundingRect(lf.contour)
            x2, y2, w2, h2 = cv2.boundingRect(contour)

        bx, by, bw, bh = cv2.boundingRect(lf.contour)

        if not x1 <= x2 <= x1 + w1:
            return False

        return (y1 <= y2 <= y1 + h1) or (y2 <= y1 <= y2 + h2)

    def match_boxoverlap(self, contour, frame):
        lf = self.frames[self.last_frame]
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # if it doesn't - check previous frame extending the ROI to the left a
        # %
        bx, by, bw, bh = cv2.boundingRect(lf.contour)
        offset_x = int(video_width / boxwindow) * int(frame - self.last_frame)
        bx = min(bx - offset_x, bx)

        if bx + bw > video_width:
            bw = video_width - bx
        else:
            bw += offset_x

        if (bx <= cx and cx <= (bx + bw) and by <= cy and cy <= (by + bh)):
            return True
        else:
            return False

    def get_last_xy(self):
        lf = self.frames[self.last_frame]
        return (lf.cx, lf.cy)

    def get_last_contour(self):
        lf = self.frames[self.last_frame]
        return lf.contour

    def get_avg_xy(self):
        avg_x = np.average([cp.cx for cp in self.frames.values()])
        avg_y = np.average([cp.cy for cp in self.frames.values()])
        return (avg_x, avg_y)

    def measure(self):

        frames = sorted(self.frames)

        if len(frames) > 3:

            first_frame = min(frames)
            last_frame = max(frames)

            fx = self.frames[first_frame].cx
            fy = self.frames[first_frame].cy
            lx = self.frames[last_frame].cx
            ly = self.frames[last_frame].cy
            num_frames = len(frames)

            avg_radius = np.average(
                [cp.radius for cp in self.frames.values()])
            avg_width = np.average(
                [cp.width for cp in self.frames.values()])
            avg_height = np.average(
                [cp.height for cp in self.frames.values()])
            avg_area = np.average([cp.area for cp in self.frames.values()])

            return (
                (num_frames, fx, fy, lx, ly, first_frame, last_frame,
                 avg_radius, avg_width, avg_height, avg_area)
            )


class ObjectDatabase(object):

    def __init__(self):
        self.prev_OverlapMovingObjects = []
        self.last_frame = -1

    def remove_MovingObjects(self, frame):

        lost_mc = [
            mc for mc in self.prev_OverlapMovingObjects if mc.last_frame < frame]
        new_mc = [
            mc for mc in self.prev_OverlapMovingObjects if mc.last_frame >= frame and len(
                mc.frames) == 1]
        existing_mc = [
            mc for mc in self.prev_OverlapMovingObjects if mc.last_frame <= frame and len(
                mc.frames) > 1]

        logging.debug("New MC: " + str(len(new_mc)) +
                      "\t combining with Lost: " + str(len(lost_mc)))

        dont_track = []
        for new in sorted(new_mc, key=lambda mc: mc.get_last_xy()[0]):
            (new_x, new_y) = new.get_last_xy()
            closest_mc = None
            closest_mc_dist = 500
            cnt = 0
            for lost in lost_mc:
                (lost_x, lost_y) = lost.get_last_xy()
                (lost_x, lost_y) = lost.get_avg_xy()
                if lost_y < new_y + 5 and lost_y > new_y - 5 and new_x < lost_x:
                    dist = lost_x - new_x
                    cnt += 1
                    if dist < closest_mc_dist:
                        closest_mc = lost
                        closest_mc_dist = dist
            if closest_mc:
                logging.debug(
                    "Found Possible New Instance Stitches: " + str(cnt))
                logging.debug("CREATED NEW INSTANCE STITCH")
                closest_mc.add_frame(new.last_frame, new.get_last_contour())
                dont_track.append(new)

        buffer = 7

        measure = [
            mc for mc in self.prev_OverlapMovingObjects if mc.last_frame < frame -
            buffer and mc not in dont_track]

        d = defaultdict(list)
        for m in measure:

            x = m.measure()
            if x:
                (num_frames, fx, fy, lx, ly, first_frame, last_frame,
                 avg_radius, avg_width, avg_height, avg_area) = m.measure()
                k = (lx, ly)
                v = (fx, m)
                d[k].append(v)

        for k in d.keys():
            (cx, m) = max(d[k], key=lambda x: x[0])
            cw.writerow(m.measure())

        self.prev_OverlapMovingObjects = [
            mc for mc in self.prev_OverlapMovingObjects if mc.last_frame >= frame -
            buffer and mc not in dont_track]

    def add_contours(self, frame, orig_contours):

        order_contours = []
        for contour in orig_contours:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if int(opts.objectdiametermin) < (radius * 2):
                order_contours.append((cx, contour))

        if not self.prev_OverlapMovingObjects:
            logging.debug(
                "Empty Moving Objects - Adding: " + str(len(order_contours)))
            for (cx, cnt) in order_contours:
                moving_object = MovingObject()
                moving_object.add_frame(frame, cnt)
                self.prev_OverlapMovingObjects.append(moving_object)
            self.remove_MovingObjects(frame)
        else:
            # Iterate over thew new contours and find one that overlaps
            logging.debug("Checking New Overlap Contour: " + str(len(order_contours))
                          + " against MC: " + str(len(self.prev_OverlapMovingObjects)))
            for (cx, cnt) in sorted(order_contours, key=lambda tup: tup[0], reverse=False):

                matching_objects = [
                    mc for mc in self.prev_OverlapMovingObjects if mc.match_overlap(
                        cnt)]

                # How to handle muliple mathes?
                if not matching_objects:
                    logging.debug("\tAdding New Instance")
                    moving_object = MovingObject()
                    moving_object.add_frame(frame, cnt)
                    self.prev_OverlapMovingObjects.append(moving_object)
                    self.remove_MovingObjects(frame)

            for mc in sorted(self.prev_OverlapMovingObjects, key=lambda x: x.get_last_xy()[0], reverse=False):

                matching_cnts = sorted(
                    [(cx,
                      cnt) for (cx,
                                cnt) in order_contours if mc.match_overlap(
                        cnt)],
                    key=lambda t: t[0],
                    reverse=True)

                if matching_cnts:
                    if len(matching_cnts) > 1:

                        for (cx, cnt) in sorted(matching_cnts, key=lambda x: x[0], reverse=False):
                            mc.add_frame(frame, cnt)

                    else:
                        logging.debug("\tMatched MovingObject to One Contour")
                        (cx, cnt) = matching_cnts[0]
                        mc.add_frame(frame, cnt)

        self.remove_MovingObjects(frame)
        self.last_frame = frame
        return


def init_logger(opts):
    level = logging.INFO
    handler = logging.StreamHandler()

    if opts.debug:
        level = logging.DEBUG
        handler = logging.StreamHandler()
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def gui_available():
    """
    Probe whether cv2's HighGUI backend can actually open a window.
    opencv-python-headless (the wheel this fork now depends on, since it's
    the one that reliably installs in Colab/CI/containers) has no HighGUI
    support at all, so cv2.namedWindow raises cv2.error; a regular
    opencv-python build with no $DISPLAY (e.g. this sandbox) fails the same
    way. Either case means -v/--visualize's interactive window, trackbars
    and waitKey() can't be used, and must be downgraded instead of crashing.
    """
    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_probe__")
        return True
    except cv2.error:
        return False


def parse_args(argv):
    if argv is None:
        argv = sys.argv[1:]
    p = optparse.OptionParser()
    p.usage = '%prog -m movie.avi'

    p.add_option(
        '-d',
        '--debug',
        action='store_true',
        default=False,
        help="debug mode")
    p.add_option('-m', '--movie', help='movie.avi')
    p.add_option('-b', '--blur', help='blur setting')
    p.add_option('-t', '--threshold', help='threshold')
    p.add_option('-c', '--objectdiametermin', help='object minimum diameter')
    p.add_option('-x', '--boxwindow', help='box window size')
    p.add_option('-p', '--playbackspeed', help='playback speed')
    p.add_option(
        '-v',
        '--visualize',
        action='store_true',
        help='visualize tracking (interactive window; falls back to inline '
             'preview frames in Colab, or is disabled with a warning if no '
             'display/GUI backend is available at all)')
    p.add_option(
        '-s',
        '--subtractbg',
        action='store_true',
        help='subtract background')
    p.add_option('-a', '--accumulator', help='accumulator weight')
    p.add_option('-o', '--output', help='output avi')
    p.add_option(
        '--colab_preview_stride',
        type='int',
        default=30,
        help='when -v is used but no interactive GUI window is available '
             '(Colab, this sandbox, etc.), show every Nth annotated frame '
             'inline instead of every frame, to avoid flooding notebook '
             'output (default: 30)')

    opts, args = p.parse_args(argv)
    # sanity check
    if not opts.movie:
        p.error("missing --movie input movie")

    if not opts.objectdiametermin:
        p.error("missing -c object diameter mininum")

    return opts, args


def main(argv=None):

    global opts
    opts, args = parse_args(argv)
    init_logger(opts)
    logging.debug(opts)
    logging.debug(args)

    global cw
    cw = csv.writer(sys.stdout, lineterminator='\n')

    capture = cv2.VideoCapture(opts.movie)
    if not capture.isOpened():
        logging.error("Could not open movie: %s", opts.movie)
        return 1

    global video_width
    video_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    global video_height
    video_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def onChange2(val):
        global blurVal
        blurVal = val + 1 if val % 2 == 0 else val
        print("Changed Blur to " + str(blurVal))

    def onChange3(val):
        global threshVal
        threshVal = val
        print("Changed Treshold to " + str(threshVal))

    global blurVal
    blurVal = int(opts.blur) if opts.blur else 7

    global threshVal
    threshVal = int(opts.threshold) if opts.threshold else 7

    # Headless-environment detection (Colab, this sandbox, CI, ...): decide
    # once up front whether -v can drive a real interactive window, and if
    # not, whether we can at least preview frames inline via Colab's
    # cv2_imshow. See PORTING_NOTES.md for why this is necessary.
    in_colab = 'google.colab' in sys.modules
    has_gui = gui_available() if opts.visualize else False
    colab_imshow = None

    if opts.visualize and not has_gui:
        if in_colab:
            from google.colab.patches import cv2_imshow as colab_imshow
            logging.warning(
                "No interactive GUI window available in Colab -- showing "
                "every %dth annotated frame inline instead (no live "
                "trackbars; pass -b/-t explicitly). Use -o to save a full "
                "annotated video.", opts.colab_preview_stride)
        else:
            logging.warning(
                "No GUI/display backend available (cv2.namedWindow failed) "
                "-- disabling -v/--visualize. Use -o to save an annotated "
                "video instead.")
            opts.visualize = False

    if opts.visualize and has_gui:
        cv2.namedWindow("Image")
        cv2.createTrackbar("Blur", "Image", 1, 25, onChange2)
        cv2.createTrackbar("Threshold", "Image", 1, 255, onChange3)
        cv2.setTrackbarPos("Blur", "Image", blurVal)
        cv2.setTrackbarPos("Threshold", "Image", threshVal)

    global boxwindow
    boxwindow = int(opts.boxwindow) if opts.boxwindow else 10

    playbackspeed = int(opts.playbackspeed) if opts.playbackspeed else 100
    accumulator_weight = float(opts.accumulator) if opts.accumulator else 0.001

    frame_cnt = 0
    object_db = ObjectDatabase()
    status = True

    status, f = capture.read()
    if not status or f is None:
        logging.error("Could not read first frame from movie: %s", opts.movie)
        return 1
    frame = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    bg_avg = np.float32(frame)

    # reset capture
    capture.release()
    capture = cv2.VideoCapture(opts.movie)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_writer = cv2.VideoWriter()

    if opts.output:
        output_writer.open(
            opts.output,
            fourcc,
            21.37,
            (int(video_width),
             int(video_height)),
            True)

    while status:

        status, img = capture.read()
        if not status or img is None:
            break
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if opts.subtractbg:
            cv2.accumulateWeighted(frame, bg_avg, accumulator_weight)

        res1 = frame
        if opts.subtractbg:
            res1 = cv2.absdiff(frame, cv2.convertScaleAbs(bg_avg))

        blurred = cv2.GaussianBlur(res1, (blurVal, blurVal), 0)
        ret, thresh = cv2.threshold(blurred, threshVal, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logging.debug("\n\nCalling Add_Contours")
        object_db.add_contours(frame_cnt, contours)

        for mc in reversed(object_db.prev_OverlapMovingObjects):
            if len(list(mc.frames.keys())) > 0:
                for f in mc.frames.keys():
                    cp = mc.frames[f]
                    cv2.circle(
                        img,
                        (int(cp.cx),
                         int(cp.cy)),
                        int(cp.radius),
                        mc.color,
                        1)
            if len(list(mc.frames.keys())) == 1:
                cv2.drawContours(
                    img, [mc.get_last_contour()], -1, (0, 255, 0), -1, 1)

        if opts.visualize:
            if has_gui:
                cv2.imshow("Image", img)
                c = cv2.waitKey(playbackspeed)
                if c == 27:  # Break if user enters 'A'.
                    frame_cnt += 1
                    continue
                if c == 97:
                    break
            elif colab_imshow is not None and frame_cnt % opts.colab_preview_stride == 0:
                colab_imshow(img)

        if opts.output:
            output_writer.write(img)

        frame_cnt += 1

    if opts.output:
        output_writer.release()

    if has_gui:
        cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    rval = main()
    logging.shutdown()
    sys.exit(rval)
