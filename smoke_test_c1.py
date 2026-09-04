"""
Smoke test for Workstream C1 deliverables.
Creates a dummy image and runs both signals to verify end-to-end output
before Hour 10 integration.
"""
import numpy as np

from signals.exif_blur_gate import analyze as blur_analyze
from signals.synthetic_vit_signal import analyze as vit_analyze


def main():
    # ── create a dummy 500x500 BGR image with random noise ──────────────────
    rng = np.random.RandomState(seed=42)
    dummy_image = rng.randint(0, 256, (500, 500, 3), dtype=np.uint8)

    print("=" * 60)
    print("  C1 SMOKE TEST - pre-integration sanity check")
    print("=" * 60)

    # -- 1. EXIF / Blur Gate --
    print("\n>> exif_blur_gate.analyze(dummy_image)")
    blur_result = blur_analyze(dummy_image)
    print(f"  signal_name   : {blur_result.signal_name}")
    print(f"  score         : {blur_result.score}")
    print(f"  applicable    : {blur_result.applicable}")
    print(f"  evidence_image: {type(blur_result.evidence_image).__name__}")
    print(f"  note          : {blur_result.note}")

    # -- 2. Synthetic ViT Signal --
    print("\n>> synthetic_vit_signal.analyze(dummy_image)")
    vit_result = vit_analyze(dummy_image)
    print(f"  signal_name   : {vit_result.signal_name}")
    print(f"  score         : {vit_result.score}")
    print(f"  applicable    : {vit_result.applicable}")
    print(f"  evidence_image: {type(vit_result.evidence_image).__name__}")
    print(f"  note          : {vit_result.note}")

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = all([
        blur_result.signal_name == "exif_blur_gate",
        vit_result.signal_name == "synthetic_generation_vit",
        isinstance(blur_result.applicable, bool),
        isinstance(vit_result.applicable, bool),
        0.0 <= blur_result.score <= 1.0,
        0.0 <= vit_result.score <= 1.0,
    ])
    print(f"  ALL CHECKS PASSED: {passed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
