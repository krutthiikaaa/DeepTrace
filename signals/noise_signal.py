"""
Workstream B — Noise consistency signal.
Compares high-frequency sensor-noise residual between the face region and the
background/border region to catch face-swap composites.
"""
import numpy as np

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    raise NotImplementedError("Workstream B: implement noise-consistency analysis")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
