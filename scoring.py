"""
Workstream C2 — verdict aggregation.
Maps SignalResults into category scores (ai_generated / face_swapped / edited)
and applies the verdict logic described in docs/PLAN.md.
"""
import logging

from shared.types import SignalResult

logger = logging.getLogger(__name__)

GENUINE = "Genuine"
AI_GENERATED = "AI-Generated"
FACE_SWAPPED = "Face-Swapped"
EDITED = "Edited"
UNCERTAIN = "Uncertain - Route to Tier-2 Review"

# Which fraud category each signal's score feeds into. Keyed by the *actual*
# signal_name each teammate's analyze() returns (confirmed against the real
# merged code on main, not just the PRD's config schema - A and B's names
# diverged from "fft"/"ela"/"pupil"/"noise"). The quality-gate signal
# (exif_blur_gate) is deliberately absent - it gates trust in the other
# signals rather than contributing to a fraud category itself.
SIGNAL_CATEGORY_MAP = {
    "FFT Artifacts": "ai_generated",              # signals/fft_signal.py
    "synthetic_generation_vit": "ai_generated",   # signals/synthetic_vit_signal.py
    "pupil_specular_mismatch": "face_swapped",    # signals/pupil_signal.py
    "sensor_noise_consistency": "face_swapped",   # signals/noise_signal.py
    "Error Level Analysis": "edited",             # signals/ela_signal.py
}

CATEGORY_LABELS = {
    "ai_generated": AI_GENERATED,
    "face_swapped": FACE_SWAPPED,
    "edited": EDITED,
}

QUALITY_GATE_SIGNAL = "exif_blur_gate"

# A signal's signal_name doesn't always match the key teammates used in
# thresholds.yaml's `weights` block. Resolved here (a C2-owned file) instead
# of chasing naming drift in the shared config.
WEIGHT_KEY_ALIASES = {
    "FFT Artifacts": "fft",
    "Error Level Analysis": "ela",
    "pupil_specular_mismatch": "pupil",
    "sensor_noise_consistency": "noise",
    "synthetic_generation_vit": "synthetic_vit",
    "exif_blur_gate": "exif",
}

DEFAULT_WEIGHT = 1.0
MIN_APPLICABLE_SIGNALS = 2


def _get_weight(signal_name: str, weights_cfg: dict) -> float:
    weights_cfg = weights_cfg or {}
    key = WEIGHT_KEY_ALIASES.get(signal_name, signal_name)
    return weights_cfg.get(key, DEFAULT_WEIGHT)


def _category_scores(category_results: list[SignalResult], weights_cfg: dict) -> dict[str, float]:
    buckets: dict[str, list[SignalResult]] = {}
    for result in category_results:
        if not result.applicable:
            continue
        category = SIGNAL_CATEGORY_MAP.get(result.signal_name)
        if category is None:
            logger.warning("Unrecognized signal_name %r - ignored by scoring", result.signal_name)
            continue
        buckets.setdefault(category, []).append(result)

    scores = {}
    for category, results in buckets.items():
        total_weight = sum(_get_weight(r.signal_name, weights_cfg) for r in results)
        if total_weight <= 0:
            continue
        weighted_sum = sum(r.score * _get_weight(r.signal_name, weights_cfg) for r in results)
        scores[category] = weighted_sum / total_weight
    return scores


def aggregate(signal_results: list[SignalResult], thresholds: dict) -> dict:
    """
    - Quality gate fail, or <2 applicable category signals -> Uncertain
    - max(category) >= high_threshold and beats runner-up by >= margin -> that category
    - all categories < low_threshold -> Genuine
    - else -> Uncertain (conflicting signals)
    See docs/PLAN.md "Scoring & Verdict Logic" for the full spec.
    """
    signal_results = signal_results or []
    thresholds = thresholds or {}
    high_threshold = thresholds.get("high_threshold", 0.6)
    low_threshold = thresholds.get("low_threshold", 0.3)
    margin = thresholds.get("margin", 0.15)
    quality_gate_threshold = thresholds.get("quality_gate_threshold", 0.6)
    weights_cfg = thresholds.get("weights", {})

    quality_gate_result = next(
        (r for r in signal_results if r.signal_name == QUALITY_GATE_SIGNAL), None
    )
    category_results = [r for r in signal_results if r.signal_name != QUALITY_GATE_SIGNAL]

    quality_gate_info = {
        "applicable": bool(quality_gate_result and quality_gate_result.applicable),
        "score": quality_gate_result.score if quality_gate_result else None,
        "note": quality_gate_result.note if quality_gate_result else "quality gate signal not present",
    }

    if quality_gate_result and quality_gate_result.applicable and quality_gate_result.score >= quality_gate_threshold:
        return {
            "verdict": UNCERTAIN,
            "confidence": quality_gate_result.score,
            "category_scores": {},
            "quality_gate": quality_gate_info,
            "reasons": ["Image quality gate failed: " + quality_gate_result.note],
        }

    applicable_count = sum(1 for r in category_results if r.applicable)
    if applicable_count < MIN_APPLICABLE_SIGNALS:
        return {
            "verdict": UNCERTAIN,
            "confidence": 0.0,
            "category_scores": {},
            "quality_gate": quality_gate_info,
            "reasons": [f"Only {applicable_count} applicable signal(s) - insufficient evidence"],
        }

    category_scores = _category_scores(category_results, weights_cfg)
    if not category_scores:
        return {
            "verdict": UNCERTAIN,
            "confidence": 0.0,
            "category_scores": {},
            "quality_gate": quality_gate_info,
            "reasons": ["No category-contributing signals were applicable"],
        }

    ranked = sorted(category_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_category, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    labeled_scores = {CATEGORY_LABELS[cat]: score for cat, score in category_scores.items()}

    if top_score >= high_threshold and (top_score - runner_up_score) >= margin:
        return {
            "verdict": CATEGORY_LABELS[top_category],
            "confidence": top_score,
            "category_scores": labeled_scores,
            "quality_gate": quality_gate_info,
            "reasons": [f"{CATEGORY_LABELS[top_category]} score {top_score:.2f} clears threshold with margin"],
        }

    if all(score < low_threshold for score in category_scores.values()):
        return {
            "verdict": GENUINE,
            "confidence": 1.0 - top_score,
            "category_scores": labeled_scores,
            "quality_gate": quality_gate_info,
            "reasons": ["All category scores below low_threshold"],
        }

    return {
        "verdict": UNCERTAIN,
        "confidence": top_score,
        "category_scores": labeled_scores,
        "quality_gate": quality_gate_info,
        "reasons": ["Conflicting or inconclusive signal scores"],
    }
