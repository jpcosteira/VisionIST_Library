#!/usr/bin/env python3
"""Build test/pan.mp4, the tapnext test clip, from a photo already in the Library.

The clip is a slow pan + zoom over ``00.jpg`` (the building photo the
lightglue/opencv/features boxes use), so every point has a known, smooth motion
and no external fixture is needed:

    python test/make_test_video.py            # writes test/pan.mp4 (32 frames)

``test_tapnext.py``, ``test_tapnext_sessions.py`` and ``live_class_sim.py`` read
any ``test/*.mp4`` that is not an ``output*`` file.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "lightglue", "test", "00.jpg")
OUT = os.path.join(HERE, "pan.mp4")
W, H, N, FPS = 256, 192, 32, 24


def main():
    img = cv2.imread(SRC)
    if img is None:
        sys.exit(f"cannot read {SRC}")
    ih, iw = img.shape[:2]
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        t = i / (N - 1)
        scale = 0.55 + 0.15 * t                      # zoom in a little
        cw, ch = int(iw * scale), int(ih * scale)
        x0 = int((iw - cw) * (0.15 + 0.5 * t))       # pan right
        y0 = int((ih - ch) * (0.2 + 0.3 * t))        # and down
        crop = img[y0:y0 + ch, x0:x0 + cw]
        writer.write(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA))
    writer.release()
    print(f"wrote {OUT}: {N} frames {W}x{H}, {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    main()
