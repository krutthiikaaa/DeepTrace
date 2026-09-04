"""
Workstream A — Error Level Analysis (ELA) signal.
Re-saves the image at a fixed JPEG quality and diffs against the original to
surface localized recompression artifacts from splicing/editing.
"""
import numpy as np

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    raise NotImplementedError("Workstream A: implement ELA analysis")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
