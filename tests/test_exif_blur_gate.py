"""Workstream C1 — tests for signals/exif_blur_gate.py."""
import numpy as np

from signals.exif_blur_gate import analyze, BLUR_THRESHOLD
from shared.types import SignalResult


def _make_flat_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Completely uniform image → VoL ≈ 0 → blurry."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _make_noisy_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Random noise → very high VoL → sharp-ish."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestBlurGate:
    """Blur detection via variance of Laplacian."""

    def test_flat_image_is_rejected(self):
        result = analyze(_make_flat_image())
        assert isinstance(result, SignalResult)
        assert result.applicable is False
        assert result.signal_name == "exif_blur_gate"

    def test_noisy_image_is_accepted(self):
        result = analyze(_make_noisy_image())
        assert isinstance(result, SignalResult)
        assert result.applicable is True
        assert 0.0 <= result.score <= 1.0

    def test_returns_signal_result_type(self):
        result = analyze(_make_flat_image())
        assert isinstance(result, SignalResult)

    def test_no_path_gives_moderate_risk(self):
        """Without image_path the EXIF check is skipped but it still works."""
        result = analyze(_make_noisy_image(), image_path=None)
        assert result.applicable is True
        assert result.score == 0.5  # default when path is None
        assert "skipped" in result.note.lower()


class TestExifRisk:
    """EXIF metadata risk scoring."""

    def test_nonexistent_path_does_not_crash(self):
        result = analyze(_make_noisy_image(), image_path="/nonexistent/image.jpg")
        assert isinstance(result, SignalResult)
        assert result.applicable is True
        # Should still return a risk score (EXIF extraction fails gracefully)
        assert 0.0 <= result.score <= 1.0
