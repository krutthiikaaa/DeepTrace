"""Workstream C2 — tests for pipeline.py orchestration."""
import cv2
import numpy as np

import pipeline
from shared.types import SignalResult
from signals import (
    ela_signal,
    exif_blur_gate,
    fft_signal,
    noise_signal,
    pupil_signal,
    synthetic_vit_signal,
)

# Real signal_name strings as returned by each teammate's analyze() (confirmed
# against the merged signals/*.py on main).
FFT = "FFT Artifacts"
ELA = "Error Level Analysis"
PUPIL = "pupil_specular_mismatch"
NOISE = "sensor_noise_consistency"
SYNTHETIC_VIT = "synthetic_generation_vit"
EXIF = "exif_blur_gate"


def fake_result(name, score=0.1, applicable=True):
    return SignalResult(signal_name=name, score=score, applicable=applicable, evidence_image=None, note="fake")


def _patch_all_signals(monkeypatch, exif_analyze=None):
    monkeypatch.setattr(fft_signal, "analyze", lambda image: fake_result(FFT))
    monkeypatch.setattr(ela_signal, "analyze", lambda image: fake_result(ELA))
    monkeypatch.setattr(pupil_signal, "analyze", lambda image: fake_result(PUPIL))
    monkeypatch.setattr(noise_signal, "analyze", lambda image: fake_result(NOISE))
    monkeypatch.setattr(synthetic_vit_signal, "analyze", lambda image: fake_result(SYNTHETIC_VIT))
    monkeypatch.setattr(
        exif_blur_gate,
        "analyze",
        exif_analyze or (lambda image, image_path=None: fake_result(EXIF, score=0.1)),
    )


def make_test_image(tmp_path):
    path = tmp_path / "sample.jpg"
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_run_end_to_end_with_mocked_signals(monkeypatch, tmp_path):
    _patch_all_signals(monkeypatch)
    image_path = make_test_image(tmp_path)

    result = pipeline.run(image_path)

    assert "verdict" in result
    assert len(result["signals"]) == 6
    names = {s["signal_name"] for s in result["signals"]}
    assert names == {FFT, ELA, PUPIL, NOISE, SYNTHETIC_VIT, EXIF}


def test_missing_image_path_returns_uncertain_without_raising(monkeypatch, tmp_path):
    _patch_all_signals(monkeypatch)
    result = pipeline.run(str(tmp_path / "does_not_exist.jpg"))
    assert result["verdict"] == "Uncertain - Route to Tier-2 Review"


def test_one_signal_raising_does_not_crash_run(monkeypatch, tmp_path):
    def boom(image):
        raise RuntimeError("simulated signal crash")

    _patch_all_signals(monkeypatch)
    monkeypatch.setattr(fft_signal, "analyze", boom)
    image_path = make_test_image(tmp_path)

    result = pipeline.run(image_path)

    fft_entry = next(s for s in result["signals"] if s["signal_name"] == FFT)
    assert fft_entry["applicable"] is False
    assert "simulated signal crash" in fft_entry["note"]


def test_exif_blur_gate_two_arg_signature_is_called_with_image_path(monkeypatch, tmp_path):
    captured = {}

    def analyze_two_arg(image, image_path=None):
        captured["image_path"] = image_path
        return fake_result(EXIF, score=0.1)

    _patch_all_signals(monkeypatch, exif_analyze=analyze_two_arg)
    image_path = make_test_image(tmp_path)

    pipeline.run(image_path)

    assert captured["image_path"] == image_path


def test_exif_blur_gate_one_arg_fallback_signature(monkeypatch, tmp_path):
    def analyze_one_arg(image):
        return fake_result(EXIF, score=0.1)

    _patch_all_signals(monkeypatch, exif_analyze=analyze_one_arg)
    image_path = make_test_image(tmp_path)

    result = pipeline.run(image_path)

    exif_entry = next(s for s in result["signals"] if s["signal_name"] == EXIF)
    assert exif_entry["applicable"] is True
