"""
Workstream B — Corneal pupil reflection signal.
Uses MediaPipe Face Mesh (iris refinement) to compare the specular glint
position/size/color between the left and right eye.
Must return applicable=False (never force a score) when an eye/glint isn't
detectable - sunglasses, extreme angle, glare are expected in real KYC photos.
"""
import numpy as np

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    raise NotImplementedError("Workstream B: implement pupil/glint comparison")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
