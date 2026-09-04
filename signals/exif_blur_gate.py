"""
Workstream C1 — EXIF & blur/quality gate.
A quality gate, not a fraud-category signal: its output tells scoring.py how
much to trust the other signals, and can force Uncertain on unusable input.

Computes:
  1. Variance of Laplacian (blur metric). If below threshold → applicable=False.
  2. EXIF metadata risk — missing critical fields (Make, Model, Software,
     DateTime) each add risk; a fully stripped EXIF is the most suspicious.
"""
import cv2
import numpy as np
from typing import Optional, Tuple

from shared.types import SignalResult

# ── tunables ────────────────────────────────────────────────────────────────
BLUR_THRESHOLD = 100.0          # VoL below this → "too blurry to analyse"
_CRITICAL_EXIF_TAGS = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "ExifImageWidth", "ExifImageHeight",
}


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """Higher value → sharper image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exif_risk(image_path: Optional[str]) -> Tuple[float, str]:
    """Return (risk_score 0‑1, human note) based on EXIF completeness.

    If no path is supplied we can't inspect EXIF, so return a moderate risk
    with an explanatory note rather than crashing.
    """
    if image_path is None:
        return 0.5, "no file path provided — EXIF check skipped"

    try:
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        pil_img = PILImage.open(image_path)
        raw_exif = pil_img._getexif()
    except Exception:
        return 0.7, "EXIF extraction failed (corrupt or unsupported format)"

    if raw_exif is None:
        return 0.9, "EXIF completely absent — metadata likely stripped"

    decoded = {TAGS.get(k, k) for k in raw_exif}
    missing = _CRITICAL_EXIF_TAGS - decoded
    if not missing:
        return 0.1, "all critical EXIF fields present"

    ratio = len(missing) / len(_CRITICAL_EXIF_TAGS)
    note = f"missing EXIF fields: {', '.join(sorted(missing))}"
    # scale: 0.2 (one field missing) … 0.9 (all missing)
    return round(0.2 + 0.7 * ratio, 3), note


def analyze(image: np.ndarray, image_path: Optional[str] = None) -> SignalResult:
    """Pure function: image (BGR np.ndarray) → SignalResult.

    Parameters
    ----------
    image : np.ndarray
        BGR image as decoded by cv2.imread.
    image_path : str, optional
        Path to the original file so EXIF can be inspected.
    """
    # ── blur check ──────────────────────────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vol = _variance_of_laplacian(gray)

    if vol < BLUR_THRESHOLD:
        return SignalResult(
            signal_name="exif_blur_gate",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note=f"image too blurry (VoL={vol:.1f} < {BLUR_THRESHOLD}), "
                 "downstream signals unreliable",
        )

    # ── EXIF risk ───────────────────────────────────────────────────────────
    risk, exif_note = _exif_risk(image_path)

    return SignalResult(
        signal_name="exif_blur_gate",
        score=risk,
        applicable=True,
        evidence_image=None,
        note=f"VoL={vol:.1f}; {exif_note}",
    )


if __name__ == "__main__":
    import sys

    img = cv2.imread(sys.argv[1])
    print(analyze(img, image_path=sys.argv[1]))
