"""
Workstream A — FFT frequency-spectrum signal.
Detects periodic checkerboard artifacts characteristic of GAN upsampling layers.
"""
import numpy as np

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    raise NotImplementedError("Workstream A: implement FFT spectrum analysis")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
