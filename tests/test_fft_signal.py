import numpy as np  # pyrefly: ignore [missing-import]
import cv2  # pyrefly: ignore [missing-import]
from signals.fft_signal import analyze

def test_fft_detects_checkerboard():
    # Synthetic Clean Image: 256x256x3 random noise heavily blurred
    clean = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    clean = cv2.GaussianBlur(clean, (15, 15), 0)
    
    # Synthetic GAN Image: inject periodic repeating checkerboard grid
    fake = clean.copy().astype(np.int16)
    fake[::4, ::4] += 30
    fake = np.clip(fake, 0, 255).astype(np.uint8)
    
    res_clean = analyze(clean)
    res_fake = analyze(fake)
    
    assert res_clean.applicable is True
    assert res_fake.applicable is True
    assert res_fake.score > res_clean.score
    assert isinstance(res_fake.evidence_image, np.ndarray)
    assert len(res_fake.evidence_image.shape) == 3

def test_fft_graceful_degradation():
    invalid = np.array([1, 2, 3])
    res = analyze(invalid)

    assert res.applicable is False
    assert res.score == 0.0
