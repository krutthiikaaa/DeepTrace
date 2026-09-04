"""
Workstream A — Error Level Analysis (ELA) signal.
Detects localized recompression artifacts to find splices and tampering.
"""
import cv2  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]

from shared.types import SignalResult


def analyze(image: np.ndarray) -> SignalResult:
    """Pure function: image (BGR, np.ndarray) -> SignalResult. No side effects."""
    try:
        if not isinstance(image, np.ndarray) or len(image.shape) < 2:
            raise ValueError("Invalid image format")

        # 1. In-Memory Recompression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if not result:
            raise ValueError("Failed to encode image to JPG")
            
        recompressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        if recompressed is None:
            raise ValueError("Failed to decode recompressed JPG")
            
        # 2. Absolute Difference
        diff = cv2.absdiff(image, recompressed)
        
        # 3. Amplify
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        scale_factor = 15.0
        amplified_gray = np.clip(diff_gray.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)
        
        # 4. Scoring via Connected Components
        # Apply a binary threshold to isolate "hot" pixels
        _, thresh = cv2.threshold(amplified_gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find distinct blobs of high error
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        score = 0.0
        if num_labels > 1:
            # Exclude the background (label 0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = np.max(areas)
            
            # Spliced regions show up as localized hot blobs, whereas genuine images have uniform global noise.
            if len(areas) > 1:
                avg_area = np.mean(areas)
                # If the max area is significantly larger than the average area, we likely have a localized anomaly
                ratio = max_area / (avg_area + 1e-5)
                # Map ratio to 0.0 - 1.0 (clamping at a ratio of 50 for max score)
                score = float(np.clip(ratio / 50.0, 0.0, 1.0))
            else:
                # If there is only one blob, score based on how substantial it is relative to the image
                img_area = image.shape[0] * image.shape[1]
                # If the blob covers more than 1% of the image, give it a high score
                score = float(np.clip(max_area / (img_area * 0.01 + 1), 0.0, 1.0))
        
        # 5. Evidence Image
        evidence_image = cv2.applyColorMap(amplified_gray, cv2.COLORMAP_INFERNO)
        
        return SignalResult(
            signal_name="Error Level Analysis",
            score=score,
            applicable=True,
            evidence_image=evidence_image,
            note=f"ELA localized anomaly score: {score:.2f}"
        )
        
    except Exception as e:
        return SignalResult(
            signal_name="Error Level Analysis",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note=f"Failed to process: {str(e)}"
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            res = analyze(img)
            print(res)
            if res.evidence_image is not None:
                cv2.imwrite("ela_evidence.jpg", res.evidence_image)
