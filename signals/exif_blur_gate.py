"""
Workstream C1 — EXIF & blur/quality gate.
A quality gate, not a fraud-category signal: its output tells scoring.py how
much to trust the other signals, and can force Uncertain on unusable input.

NOTE: EXIF metadata is stripped once an image is decoded into a plain
np.ndarray (e.g. via cv2.imread). If EXIF inspection is needed, this module
may need the original file path/bytes in addition to the decoded array -
flag this to the group during the hour 0-1 contract sync if so, since the
frozen SignalResult contract currently assumes `analyze(image) -> SignalResult`
for every signal.
"""
import numpy as np

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    raise NotImplementedError("Workstream C1: implement EXIF/blur/resolution quality gate")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
