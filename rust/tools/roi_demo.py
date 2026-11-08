#!/usr/bin/env python3
"""Highlight the regions of interest the monitor looks at, on a camera frame.

Mirrors the Rust pipeline (src/align.rs, src/recognise.rs): registers the frame
onto the dataset framing with SIFT + RANSAC, finds the display and the phase
bar by template matching, splits the bar into six bands and applies the
phase rule. Everything is drawn back onto the camera's own view, with zoomed
insets of the aligned display and bar. The digits are not read (that needs the
digit model; use `washing-machine-monitor recognise` for the full reading).

Requires opencv-python(-headless) and numpy.

Usage:
  python3 roi_demo.py FRAME.jpg [-o out.png]
  python3 roi_demo.py 'rtsp://user:pass@camera:554/stream' [-o out.png]
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
REFERENCE = os.path.join(REPO, 'rust', 'assets', 'reference_frame.jpg')
DISPLAY_TEMPLATE = os.path.join(REPO, 'recognise', 'display_template.png')
BAR_TEMPLATE = os.path.join(REPO, 'recognise', 'template_bar.jpg')

# Keep in sync with src/align.rs and src/recognise.rs.
SIZE = (1280, 720)
PANEL = np.array([(250, 205), (1150, 265), (1100, 595), (440, 415), (255, 335)], np.float32)
MATCH_RATIO, RANSAC_THRESHOLD, MIN_INLIERS, ALIGNED_TOLERANCE = 0.75, 4.0, 25, 1.0
BAR_SEARCH_OFFSET_X, BAR_SEARCH_WIDTH = 300, 250
DIGIT_CELLS = [(0, 21), (18, 14), (30, 16)]
BAR_THRESHOLD_K, BAR_OPENING_SIZE, BAR_LIT_MAX_MEAN, BAR_LIT_MIN_EXCESS = 460.0, 6, 20.0, 20.0
PHASES = ['Ammollo/Prelavaggio', 'Lavaggio', 'Risciacquo', 'Stop con acqua',
          'Scarico/Centrifuga', 'Antipiega/Fine']

# BGR colours.
PANEL_C, DISPLAY_C, CELL_C, BAR_C, BAND_C, LIT_C = (
    (255, 160, 0), (0, 220, 0), (0, 230, 255), (255, 255, 0), (160, 160, 160), (0, 0, 255))


def load_frame(source):
    if '://' not in source:
        frame = cv2.imread(source, cv2.IMREAD_COLOR)
        if frame is None:
            sys.exit(f'cannot read {source}')
        return frame
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    frame = None
    for _ in range(6):  # skip the first frames, often smeared mid-GOP
        ok, f = capture.read()
        if ok:
            frame = f
    capture.release()
    if frame is None:
        sys.exit('no frame received from the stream')
    return frame


def register(grey, reference):
    """Homography frame -> reference, number of inliers, panel shift (px)."""
    sift = cv2.SIFT_create()
    mask = np.zeros_like(reference)
    cv2.fillConvexPoly(mask, PANEL.astype(np.int32), 255)
    rkp, rdesc = sift.detectAndCompute(reference, mask)
    kp, desc = sift.detectAndCompute(grey, None)
    if desc is None or len(kp) < MIN_INLIERS:
        sys.exit('too few features in the frame')
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, rdesc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < MATCH_RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        sys.exit(f'the panel is not in view ({len(good)} matches)')
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([rkp[m.trainIdx].pt for m in good])
    h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    if h is None or inliers.sum() < MIN_INLIERS:
        sys.exit('cannot register the frame onto the reference')
    in_frame = cv2.perspectiveTransform(PANEL.reshape(-1, 1, 2), np.linalg.inv(h)).reshape(-1, 2)
    shift = float(np.linalg.norm(in_frame - PANEL, axis=1).mean())
    if shift < ALIGNED_TOLERANCE:
        h = np.eye(3)
    return h, int(inliers.sum()), len(good), shift


def find(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    return (x, y, template.shape[1], template.shape[0]), score


def regions(aligned, display_t, bar_t):
    display, display_score = find(aligned[:, :SIZE[0] // 2], display_t)
    x = display[0] + display[2] + BAR_SEARCH_OFFSET_X
    area = aligned[display[1]:, x:x + BAR_SEARCH_WIDTH].copy()
    mean = np.average(area)
    area[area > mean] = int(mean)
    (bx, by, bw, bh), bar_score = find(area, bar_t)
    return display, display_score, (bx + x, by + display[1], bw, bh), bar_score


def bands(bar):
    """Per band: stripe vanished (bool), brightness excess over the median, lit."""
    sd = float(np.std(bar))
    step = bar.shape[0] // len(PHASES)
    raw = np.array([bar[i * step:(i + 1) * step].mean() for i in range(len(PHASES))])
    excess = raw - np.median(raw)
    if sd < 1e-6:
        return step, [False] * len(PHASES), excess, None
    _, binary = cv2.threshold(bar, float(np.average(bar)) - BAR_THRESHOLD_K / sd, 255,
                              cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (BAR_OPENING_SIZE, BAR_OPENING_SIZE))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    vanished = [opened[i * step:(i + 1) * step].mean() <= BAR_LIT_MAX_MEAN for i in range(len(PHASES))]
    lit = next((i for i in range(len(PHASES)) if vanished[i] and excess[i] >= BAR_LIT_MIN_EXCESS), None)
    return step, vanished, excess, lit


def to_frame(points, h_inv):
    pts = np.float32(points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2).astype(np.int32)


def rect_points(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def label(img, text, org, colour, scale=0.5, thickness=1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def inset(img, crop, scale, org, title):
    zoom = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    zoom = cv2.cvtColor(zoom, cv2.COLOR_GRAY2BGR)
    x, y = org
    img[y:y + zoom.shape[0], x:x + zoom.shape[1]] = zoom
    cv2.rectangle(img, (x - 1, y - 1), (x + zoom.shape[1], y + zoom.shape[0]), (255, 255, 255), 1)
    label(img, title, (x, y - 6), (255, 255, 255), 0.45)
    return zoom


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='image file or stream URL')
    parser.add_argument('-o', '--output', default='roi_demo.png')
    args = parser.parse_args()

    frame = cv2.resize(load_frame(args.source), SIZE, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)
    display_t = cv2.imread(DISPLAY_TEMPLATE, cv2.IMREAD_GRAYSCALE)
    bar_t = cv2.imread(BAR_TEMPLATE, cv2.IMREAD_GRAYSCALE)

    h, inliers, matches, shift = register(grey, reference)
    aligned = grey if shift < ALIGNED_TOLERANCE else cv2.warpPerspective(grey, h, SIZE, flags=cv2.INTER_CUBIC)
    display, display_score, bar, bar_score = regions(aligned, display_t, bar_t)
    bx, by, bw, bh = bar
    step, vanished, excess, lit = bands(aligned[by:by + bh, bx:bx + bw])
    h_inv = np.linalg.inv(h)

    out = frame.copy()
    # Panel used for registration.
    cv2.polylines(out, [to_frame(PANEL, h_inv)], True, PANEL_C, 2, cv2.LINE_AA)
    # Display and digit cells.
    dx, dy, dw, dh = display
    cv2.polylines(out, [to_frame(rect_points(*display), h_inv)], True, DISPLAY_C, 2, cv2.LINE_AA)
    for i, (cx, cw) in enumerate(DIGIT_CELLS):
        cv2.polylines(out, [to_frame(rect_points(dx + cx, dy, cw, dh), h_inv)], True, CELL_C, 1, cv2.LINE_AA)
    top_left = to_frame([(dx, dy)], h_inv)[0]
    label(out, f'display ({display_score:.2f})', (top_left[0], top_left[1] - 8), DISPLAY_C)
    # Phase bar and its bands.
    cv2.polylines(out, [to_frame(rect_points(*bar), h_inv)], True, BAR_C, 2, cv2.LINE_AA)
    for i, name in enumerate(PHASES):
        band = to_frame(rect_points(bx, by + i * step, bw, step), h_inv)
        if i == lit:
            overlay = out.copy()
            cv2.fillPoly(overlay, [band], LIT_C)
            out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
        cv2.polylines(out, [band], True, LIT_C if i == lit else BAND_C, 1, cv2.LINE_AA)
    bar_top = to_frame([(bx, by)], h_inv)[0]
    label(out, f'phase bar ({bar_score:.2f})', (bar_top[0] - 40, bar_top[1] - 8), BAR_C)

    # Summary.
    state = f'RUNNING - {PHASES[lit]}' if lit is not None else 'OFF (no phase lit)'
    label(out, state, (12, 30), (0, 0, 255) if lit is not None else (0, 255, 0), 0.8, 2)
    label(out, f'camera shift {shift:.0f}px, {inliers}/{matches} matches '
               f'({"warped" if shift >= ALIGNED_TOLERANCE else "already aligned"})', (12, 58), PANEL_C)

    # Insets of the aligned display and bar, as the pipeline sees them.
    x0 = SIZE[0] - 480
    zoom = inset(out, aligned[dy:dy + dh, dx:dx + dw], 4, (x0, 30), 'display (aligned)')
    for cx, cw in DIGIT_CELLS:
        cv2.rectangle(out, (x0 + cx * 4, 30), (x0 + (cx + cw) * 4, 30 + dh * 4), CELL_C, 1)
    bx0 = x0 + zoom.shape[1] + 30
    inset(out, aligned[by:by + bh, bx:bx + bw], 3, (bx0, 30), 'bar')
    for i, name in enumerate(PHASES):
        y = 30 + i * step * 3
        cv2.line(out, (bx0, y), (bx0 + bw * 3, y), BAND_C, 1)
        mark = '*' if i == lit else ('o' if vanished[i] else ' ')
        label(out, f'{mark} {excess[i]:+.0f} {name}', (bx0 + bw * 3 + 6, y + step * 3 // 2 + 4),
              LIT_C if i == lit else (220, 220, 220), 0.38)
    label(out, '* lit   o stripe vanished   +n brighter than median', (x0, 30 + bh * 3 + 22),
          (220, 220, 220), 0.38)

    cv2.imwrite(args.output, out)
    print(f'{state}; shift {shift:.0f}px, {inliers}/{matches} matches; '
          f'display {display_score:.2f}, bar {bar_score:.2f} -> {args.output}')


if __name__ == '__main__':
    main()
