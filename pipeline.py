"""
Workstream C2 — orchestration entrypoint.
run() is the single function every other workstream (especially Workstream D's
app.py) calls: load image -> call each signals.*.analyze(), each wrapped in
its own try/except (one bad signal must never crash the whole request) ->
scoring.aggregate() -> return result dict.
"""
import logging

import cv2
import yaml

import scoring
from shared.types import SignalResult
from signals import (
    ela_signal,
    exif_blur_gate,
    fft_signal,
    noise_signal,
    pupil_signal,
    synthetic_vit_signal,
)

logger = logging.getLogger(__name__)

SIGNAL_MODULES = [fft_signal, ela_signal, pupil_signal, noise_signal, exif_blur_gate, synthetic_vit_signal]

# The signal_name each module's analyze() returns on success - used to label
# the fallback SignalResult on failure so scoring.py still categorizes/weighs
# it correctly (module filenames don't all match their signal_name, e.g.
# synthetic_vit_signal.py returns "synthetic_generation_vit").
MODULE_SIGNAL_NAMES = {
    fft_signal: "FFT Artifacts",
    ela_signal: "Error Level Analysis",
    pupil_signal: "pupil_specular_mismatch",
    noise_signal: "sensor_noise_consistency",
    exif_blur_gate: "exif_blur_gate",
    synthetic_vit_signal: "synthetic_generation_vit",
}

THRESHOLDS_PATH = "configs/thresholds.yaml"


def _load_thresholds() -> dict:
    try:
        with open(THRESHOLDS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - fail-soft: a bad config must never crash the pipeline
        logger.warning("Could not load %s: %s - using empty thresholds", THRESHOLDS_PATH, e)
        return {}


def _run_signal(module, image, image_path: str) -> SignalResult:
    name = MODULE_SIGNAL_NAMES.get(module, module.__name__.rsplit(".", 1)[-1])
    try:
        if module is exif_blur_gate:
            try:
                return module.analyze(image, image_path=image_path)
            except TypeError:
                return module.analyze(image)
        return module.analyze(image)
    except Exception as e:  # noqa: BLE001 - fail-soft: one broken signal must never crash the request
        logger.warning("Signal %s failed: %s", name, e)
        return SignalResult(
            signal_name=name,
            score=0.0,
            applicable=False,
            evidence_image=None,
            note=f"signal error: {e}",
        )


def _uncertain_result(note: str) -> dict:
    return {
        "verdict": scoring.UNCERTAIN,
        "confidence": 0.0,
        "category_scores": {},
        "quality_gate": {"applicable": False, "score": None, "note": note},
        "reasons": [note],
        "signals": [],
    }


def run(image_path: str) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return _uncertain_result(f"Could not load image: {image_path}")

    signal_results = [_run_signal(module, image, image_path) for module in SIGNAL_MODULES]

    thresholds = _load_thresholds()
    verdict = scoring.aggregate(signal_results, thresholds)

    verdict["signals"] = [
        {
            "signal_name": r.signal_name,
            "score": r.score,
            "applicable": r.applicable,
            "note": r.note,
            "evidence_image": r.evidence_image,
        }
        for r in signal_results
    ]
    return verdict


if __name__ == "__main__":
    import pprint
    import sys

    result = run(sys.argv[1])
    pprint.pprint({k: v for k, v in result.items() if k != "signals"})
    print("\nsignals:")
    for s in result["signals"]:
        print(f"  {s['signal_name']}: score={s['score']:.2f} applicable={s['applicable']} note={s['note']}")
