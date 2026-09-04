"""
Workstream C1 (stretch) — pretrained synthetic-image detector.
Loads buildborderless/CommunityForensics-DeepfakeDet-ViT from the Hugging Face
Hub, inference-only (no fine-tuning). Must degrade gracefully (applicable=False)
if torch/transformers/timm aren't installed or the model fails to load, so its
absence never breaks the core pipeline. Pre-download the checkpoint before the
event rather than relying on a live pull.
"""
import numpy as np

from shared.types import SignalResult

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: F401

            # TODO: load "buildborderless/CommunityForensics-DeepfakeDet-ViT"
            _model_cache = False  # placeholder until implemented
        except Exception:
            _model_cache = False
    return _model_cache or None


def analyze(image: np.ndarray) -> SignalResult:
    model = _load_model()
    if model is None:
        return SignalResult(
            signal_name="synthetic_generation_vit",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note="stretch model unavailable - pipeline continues without it",
        )
    raise NotImplementedError("Workstream C1: implement ViT inference + score")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
