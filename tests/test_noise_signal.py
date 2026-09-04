import pytest
import numpy as np
import cv2
from signals.noise_signal import analyze, SignalResult

def test_noise_signal_uniform_noise():
    """Construct a synthetic 400x400 image with uniform Gaussian noise -> asserts applicable == True and score < 0.40"""
    img = np.full((400, 400, 3), 128, dtype=np.float32)
    noise = np.random.normal(0, 15, (400, 400, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    res = analyze(img)
    
    assert isinstance(res, SignalResult)
    assert res.applicable is True, f"Failed with note: {res.note}"
    assert res.score < 0.40

def test_noise_signal_mismatched_composite():
    """Construct an image where central 150x150 block is heavily smoothed while outer border has heavy Gaussian noise -> score > 0.50"""
    img = np.full((400, 400, 3), 128, dtype=np.float32)
    
    noise = np.random.normal(0, 30, (400, 400, 3))
    img = img + noise
    
    img[125:275, 125:275] = 128
    
    center = img[125:275, 125:275].astype(np.uint8)
    center_blur = cv2.GaussianBlur(center, (15, 15), 1.0)
    img[125:275, 125:275] = center_blur
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    res = analyze(img)
    
    assert isinstance(res, SignalResult)
    assert res.applicable is True
    assert res.score > 0.50

def test_noise_signal_solid_color():
    """Pass a completely uniform gray/black image -> asserts graceful handling (no zero-division crashes)"""
    img = np.full((300, 300, 3), 50, dtype=np.uint8)
    res = analyze(img)
    
    assert isinstance(res, SignalResult)

def test_noise_signal_contract_integrity():
    """Validates that returned object matches SignalResult schema and evidence image is valid when applicable"""
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    res = analyze(img)
    
    assert isinstance(res, SignalResult)
    assert isinstance(res.signal_name, str)
    assert isinstance(res.score, float)
    assert isinstance(res.applicable, bool)
    assert isinstance(res.note, str)
    
    if res.applicable and res.evidence_image is not None:
        assert isinstance(res.evidence_image, np.ndarray)
        assert len(res.evidence_image.shape) == 3
        assert res.evidence_image.shape[2] == 3
        assert res.evidence_image.dtype == np.uint8
