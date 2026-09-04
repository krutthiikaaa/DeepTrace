"""
Workstream A — FFT frequency-spectrum signal.
Detects periodic checkerboard artifacts characteristic of GAN upsampling layers.
"""
import cv2  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape

    # Compute 2D FFT and extract shifted log-magnitude spectrum
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Add a small epsilon to avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Analyze frequency domain for periodic peak energy
    # We focus on the high frequencies and exclude the vertical/horizontal axes 
    # and the DC component (center), which contain most of the natural image energy.
    cy, cx = h // 2, w // 2
    mask = np.ones((h, w), dtype=np.uint8)
    
    # Mask DC component (center)
    r = int(min(h, w) * 0.05)
    cv2.circle(mask, (cx, cy), r, 0, -1)
    
    # Mask main axes
    axis_width = max(1, int(min(h, w) * 0.01))
    mask[cy - axis_width:cy + axis_width, :] = 0
    mask[:, cx - axis_width:cx + axis_width] = 0
    
    high_freq = magnitude_spectrum * mask
    active_vals = high_freq[mask == 1]
    
    if len(active_vals) == 0:
        return SignalResult(
            signal_name="FFT Artifacts",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note="Invalid image dimensions for frequency analysis."
        )

    # Calculate score based on abnormal peak prominence in high frequencies
    median_val = np.median(active_vals)
    max_val = np.percentile(active_vals, 99.9)
    
    # Diff is the peak energy relative to median energy in high frequencies
    diff = max_val - median_val
    
    # Map to 0.0 - 1.0 (typical GAN artifacts exhibit larger diffs)
    score = float(np.clip(diff / 50.0, 0.0, 1.0))

    # Construct the evidence image
    # Normalize original spectrum to 0-255 for visual rendering
    vis_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_spectrum_bgr = cv2.cvtColor(vis_spectrum, cv2.COLOR_GRAY2BGR)
    
    # Annotate peaks
    if diff > 10.0:
        threshold = np.percentile(active_vals, 99.9)
        peaks_y, peaks_x = np.where((high_freq >= threshold) & (mask == 1))
        
        for y, x in zip(peaks_y, peaks_x):
            cv2.circle(vis_spectrum_bgr, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=1)

    return SignalResult(
        signal_name="FFT Artifacts",
        score=score,
        applicable=True,
        evidence_image=vis_spectrum_bgr,
        note=f"High-frequency peak difference: {diff:.1f}"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            print(analyze(img))
        else:
            print(f"Failed to read image: {sys.argv[1]}")
    else:
        print("Usage: python fft_signal.py <image_path>")
