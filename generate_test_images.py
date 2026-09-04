"""Temporary script to generate test images for Phase 3 evaluation."""
import os
import cv2
import numpy as np

OUTPUT_DIR = "test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Baseline sharp image — white rectangle on black background
sharp = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.rectangle(sharp, (100, 100), (400, 400), (255, 255, 255), -1)
cv2.rectangle(sharp, (150, 200), (350, 300), (0, 0, 0), -1)
cv2.line(sharp, (0, 0), (500, 500), (128, 128, 128), 3)
cv2.line(sharp, (500, 0), (0, 500), (128, 128, 128), 3)
path1 = os.path.join(OUTPUT_DIR, "1_baseline_sharp.jpg")
cv2.imwrite(path1, sharp)
print(f"[OK] {path1}")

# 2. Heavily blurred version — should fail Variance of Laplacian check
blurry = cv2.GaussianBlur(sharp, (51, 51), 0)
blurry = cv2.GaussianBlur(blurry, (51, 51), 0)  # double-blur for extra softness
path2 = os.path.join(OUTPUT_DIR, "2_reject_blurry.jpg")
cv2.imwrite(path2, blurry)
print(f"[OK] {path2}")

# 3. Pure random noise — simulates synthetic/fake texture
rng = np.random.RandomState(42)
noise = rng.randint(0, 256, (500, 500, 3), dtype=np.uint8)
path3 = os.path.join(OUTPUT_DIR, "3_synthetic_fake.jpg")
cv2.imwrite(path3, noise)
print(f"[OK] {path3}")

print("\nAll test images generated successfully.")
