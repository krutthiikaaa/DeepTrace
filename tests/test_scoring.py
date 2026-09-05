"""Workstream C2 — tests for scoring.py aggregation/verdict logic."""
from scoring import AI_GENERATED, EDITED, FACE_SWAPPED, GENUINE, UNCERTAIN, aggregate
from shared.types import SignalResult

THRESHOLDS = {
    "high_threshold": 0.6,
    "low_threshold": 0.3,
    "margin": 0.15,
    "quality_gate_threshold": 0.6,
    "weights": {"fft": 1.0, "synthetic_vit": 1.2, "pupil": 1.0, "noise": 1.0, "ela": 1.0, "exif": 0.5},
}

# Real signal_name strings as returned by each teammate's analyze() (confirmed
# against the merged signals/*.py on main - not the short names implied by
# thresholds.yaml's weights keys).
FFT = "FFT Artifacts"
ELA = "Error Level Analysis"
PUPIL = "pupil_specular_mismatch"
NOISE = "sensor_noise_consistency"
SYNTHETIC_VIT = "synthetic_generation_vit"
EXIF = "exif_blur_gate"


def sr(signal_name, score, applicable=True, note=""):
    return SignalResult(signal_name=signal_name, score=score, applicable=applicable, evidence_image=None, note=note)


def test_quality_gate_failure_forces_uncertain():
    results = [
        sr(EXIF, 0.9, applicable=True),
        sr(FFT, 0.9),
        sr(ELA, 0.9),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == UNCERTAIN
    assert result["quality_gate"]["applicable"] is True


def test_blur_quality_gate_failure_forces_uncertain():
    # exif_blur_gate can fail soft (applicable=False, e.g. too blurry) rather
    # than reporting a high EXIF-risk score. That must force Uncertain too,
    # even when other signals would otherwise clear high_threshold + margin.
    results = [
        sr(EXIF, 0.0, applicable=False, note="image too blurry (VoL=10.4 < 100.0)"),
        sr(FFT, 0.9),
        sr(ELA, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == UNCERTAIN
    assert result["quality_gate"]["applicable"] is False


def test_fewer_than_two_applicable_signals_is_uncertain():
    results = [
        sr(FFT, 0.9, applicable=True),
        sr(ELA, 0.9, applicable=False),
        sr(EXIF, 0.1, applicable=True),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == UNCERTAIN


def test_clear_high_margin_category_wins():
    results = [
        sr(FFT, 0.9),
        sr(SYNTHETIC_VIT, 0.85),
        sr(PUPIL, 0.2),
        sr(NOISE, 0.2),
        sr(EXIF, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == AI_GENERATED


def test_face_swapped_category_wins():
    results = [
        sr(PUPIL, 0.9),
        sr(NOISE, 0.85),
        sr(FFT, 0.1),
        sr(ELA, 0.1),
        sr(EXIF, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == FACE_SWAPPED


def test_edited_category_wins_with_single_applicable_signal_in_category():
    # A category needs only its own applicable signal(s) to score - the
    # >=2-applicable-signals gate is global, not per-category.
    results = [
        sr(ELA, 0.9),
        sr(FFT, 0.1),
        sr(EXIF, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == EDITED


def test_all_low_scores_is_genuine():
    results = [
        sr(FFT, 0.1),
        sr(ELA, 0.1),
        sr(PUPIL, 0.1),
        sr(NOISE, 0.1),
        sr(EXIF, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == GENUINE


def test_conflicting_mid_scores_is_uncertain():
    results = [
        sr(FFT, 0.5),
        sr(PUPIL, 0.5),
        sr(EXIF, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == UNCERTAIN


def test_unrecognized_signal_name_is_ignored_not_crashing():
    results = [
        sr(FFT, 0.9),
        sr(SYNTHETIC_VIT, 0.85),
        sr("some_future_signal", 0.99),
        sr(EXIF, 0.1),
    ]
    result = aggregate(results, THRESHOLDS)
    assert result["verdict"] == AI_GENERATED


def test_empty_signal_list_does_not_crash():
    result = aggregate([], THRESHOLDS)
    assert result["verdict"] == UNCERTAIN


def test_missing_thresholds_keys_use_defaults():
    results = [sr(FFT, 0.9), sr(PUPIL, 0.9)]
    result = aggregate(results, {})
    assert result["verdict"] in (AI_GENERATED, FACE_SWAPPED, UNCERTAIN)
