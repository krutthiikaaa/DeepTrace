"""
Workstream C1 (stretch) — pretrained synthetic-image detector.
Loads buildborderless/CommunityForensics-DeepfakeDet-ViT from the Hugging Face
Hub, inference-only (no fine-tuning). Must degrade gracefully (applicable=False)
if torch/transformers/timm aren't installed or the model fails to load, so its
absence never breaks the core pipeline. Pre-download the checkpoint before the
event rather than relying on a live pull.
"""
import numpy as np
from typing import Optional

from shared.types import SignalResult

_MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
_model_cache: Optional[object] = None
_load_failed: bool = False


def _unavailable(reason: str) -> SignalResult:
    return SignalResult(
        signal_name="synthetic_generation_vit",
        score=0.0,
        applicable=False,
        evidence_image=None,
        note=reason,
    )


def analyze(image: np.ndarray) -> SignalResult:
    """Run ViT deepfake detector on a BGR np.ndarray image.

    All heavy imports (torch, transformers, PIL) are lazy-loaded inside this
    function so the module can always be imported without those deps installed.
    Returns applicable=False if the model cannot be loaded or inference fails.
    """
    global _model_cache, _load_failed

    # ── lazy model loading (inside analyze per contract) ────────────────────
    if _load_failed:
        return _unavailable(
            "stretch model previously failed to load — skipping"
        )

    if _model_cache is None:
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoImageProcessor,
                AutoModelForImageClassification,
            )

            processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
            model = AutoModelForImageClassification.from_pretrained(_MODEL_ID)
            model.eval()
            _model_cache = (processor, model)
        except Exception:
            _load_failed = True
            return _unavailable(
                "stretch model unavailable (torch/transformers not installed "
                "or model download failed) — pipeline continues without it"
            )

    # ── inference ───────────────────────────────────────────────────────────
    try:
        import torch
        from PIL import Image as PILImage

        processor, model = _model_cache

        # Convert BGR numpy → RGB PIL (what HF processors expect)
        rgb = image[:, :, ::-1]  # BGR → RGB
        pil_img = PILImage.fromarray(rgb.astype(np.uint8))

        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

        # Convention: label-1 = "fake" / "synthetic", label-0 = "real"
        id2label = model.config.id2label
        fake_idx = None
        for idx, label in id2label.items():
            if "fake" in str(label).lower() or "synthetic" in str(label).lower():
                fake_idx = int(idx)
                break
        if fake_idx is None:
            # Fallback: assume last class is the positive (synthetic) class
            fake_idx = len(probs) - 1

        score = float(probs[fake_idx].item())

        return SignalResult(
            signal_name="synthetic_generation_vit",
            score=round(score, 4),
            applicable=True,
            evidence_image=None,
            note=f"ViT synthetic-detection confidence: {score:.2%}",
        )

    except Exception as exc:
        return _unavailable(f"inference failed: {exc}")


if __name__ == "__main__":
    import sys
    import cv2

    img = cv2.imread(sys.argv[1])
    print(analyze(img))
