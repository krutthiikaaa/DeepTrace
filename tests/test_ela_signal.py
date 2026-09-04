import numpy as np  # pyrefly: ignore [missing-import]
import cv2  # pyrefly: ignore [missing-import]
from signals.ela_signal import analyze

def test_ela_detects_splice():
    # Synthetic Clean Image: 256x256x3 pure solid color
    clean = np.full((256, 256, 3), 128, dtype=np.uint8)
    
    # Synthetic Spliced Image: solid color with a 50x50 block of dense high-frequency noise
    spliced = clean.copy()
    noise = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    # Insert the block in the center
    center_y, center_x = 128, 128
    spliced[center_y-25:center_y+25, center_x-25:center_x+25] = noise
    
    res_clean = analyze(clean)
    res_spliced = analyze(spliced)
    
    assert res_clean.applicable is True
    assert res_spliced.applicable is True
    assert res_spliced.score > res_clean.score
    assert isinstance(res_spliced.evidence_image, np.ndarray)
    assert len(res_spliced.evidence_image.shape) == 3

def test_ela_graceful_degradation():
    invalid = np.array([1, 2, 3])
    res = analyze(invalid)
    assert res.applicable is False
    assert res.score == 0.0
