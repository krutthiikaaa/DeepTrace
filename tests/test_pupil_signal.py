import pytest
import numpy as np
from signals.pupil_signal import analyze, SignalResult

def test_pupil_signal_no_face():
    """Pass a black image or random noise array -> applicable == False"""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = analyze(img)
    
    assert isinstance(res, SignalResult)
    assert res.applicable is False
    assert res.score == 0.0

def test_pupil_signal_invalid_inputs():
    """Pass 1D arrays, empty arrays, and single-channel arrays -> graceful handling"""
    img1 = np.zeros((100,), dtype=np.uint8)
    res1 = analyze(img1)
    assert isinstance(res1, SignalResult)
    assert res1.applicable is False
    
    img2 = np.zeros((0, 0, 3), dtype=np.uint8)
    res2 = analyze(img2)
    assert isinstance(res2, SignalResult)
    assert res2.applicable is False
    
    img3 = np.zeros((200, 200), dtype=np.uint8)
    res3 = analyze(img3)
    assert isinstance(res3, SignalResult)
    assert res3.applicable is False

def test_pupil_signal_contract_integrity():
    """Validates that returned object has types: signal_name is str, score is float in [0.0, 1.0], applicable is bool, note is str"""
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    res = analyze(img)
    
    assert isinstance(res, SignalResult)
    assert isinstance(res.signal_name, str)
    assert isinstance(res.score, float)
    assert 0.0 <= res.score <= 1.0
    assert isinstance(res.applicable, bool)
    assert isinstance(res.note, str)
    assert len(res.note) > 0
