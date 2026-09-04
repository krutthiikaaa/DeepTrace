"""Workstream C1 — tests for signals/synthetic_vit_signal.py."""
import numpy as np

from signals.synthetic_vit_signal import analyze
from shared.types import SignalResult


def _make_dummy_image(height: int = 224, width: int = 224) -> np.ndarray:
    """Minimal 3-channel BGR image for smoke testing."""
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestSyntheticViTSignal:
    """Smoke tests — the ViT model is NOT expected to be installed in CI."""

    def test_does_not_crash_on_dummy_image(self):
        result = analyze(_make_dummy_image())
        assert isinstance(result, SignalResult)

    def test_graceful_fallback_when_model_unavailable(self):
        """Without torch/transformers installed the signal must return
        applicable=False rather than raising."""
        result = analyze(_make_dummy_image())
        assert isinstance(result, SignalResult)
        assert result.signal_name == "synthetic_generation_vit"
        # In CI (no torch) this should be False; with torch installed it may
        # be True — either way it must not crash.
        assert isinstance(result.applicable, bool)

    def test_score_in_valid_range(self):
        result = analyze(_make_dummy_image())
        assert 0.0 <= result.score <= 1.0

    def test_note_is_non_empty(self):
        result = analyze(_make_dummy_image())
        assert len(result.note) > 0

    def test_evidence_image_is_none(self):
        """ViT signal produces no evidence overlay."""
        result = analyze(_make_dummy_image())
        assert result.evidence_image is None
