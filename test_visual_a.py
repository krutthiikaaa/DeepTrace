import cv2  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]

# Alias the analyze functions to avoid conflicts
from signals.ela_signal import analyze as analyze_ela
from signals.fft_signal import analyze as analyze_fft

def main():
    print("--- Visual Smoke Test for Workstream A ---")
    
    # =========================================================================
    # 1. Generate ELA Test Image
    # Create a 300x300 solid color and insert a noisy 50x50 block in the center
    # =========================================================================
    ela_test_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    noise_block = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    # Insert at center
    ela_test_img[125:175, 125:175] = noise_block
    
    # =========================================================================
    # 2. Generate FFT Test Image
    # Create a 300x300 blurred random noise image and overlay a periodic grid
    # =========================================================================
    fft_test_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    fft_test_img = cv2.GaussianBlur(fft_test_img, (15, 15), 0)
    
    # Cast to int16 to prevent overflow when adding intensity
    fft_test_img_float = fft_test_img.astype(np.int16)
    # Add +30 intensity to every 4th pixel (both horizontally and vertically)
    fft_test_img_float[::4, ::4] += 30
    # Clip and cast back to uint8
    fft_test_img = np.clip(fft_test_img_float, 0, 255).astype(np.uint8)
    
    # =========================================================================
    # 3. Run Analysis & Print Results
    # =========================================================================
    print("\n[Running ELA Analysis]")
    res_ela = analyze_ela(ela_test_img)
    print(f"ELA Score: {res_ela.score:.3f} | Applicable: {res_ela.applicable}")
    print(f"Note: {res_ela.note}")
    
    print("\n[Running FFT Analysis]")
    res_fft = analyze_fft(fft_test_img)
    print(f"FFT Score: {res_fft.score:.3f} | Applicable: {res_fft.applicable}")
    print(f"Note: {res_fft.note}")
    
    # =========================================================================
    # 4. Display Evidence Images
    # =========================================================================
    if res_ela.evidence_image is not None:
        cv2.imshow("ELA Heatmap (Evidence)", res_ela.evidence_image)
    else:
        print("No ELA evidence image generated.")
        
    if res_fft.evidence_image is not None:
        cv2.imshow("FFT Spectrum (Evidence)", res_fft.evidence_image)
    else:
        print("No FFT evidence image generated.")
        
    print("\nVisual windows are now open. Press any key while the window is focused to close them.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
