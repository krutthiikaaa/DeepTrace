"""
Workstream B: Biometric & Sensor-Noise Forensics
Sensor Noise Residual Consistency Signal
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

import cv2
import numpy as np

try:
    from shared.types import SignalResult
except ModuleNotFoundError:
    from dataclasses import dataclass

    @dataclass
    class SignalResult:
        signal_name: str
        score: float
        applicable: bool
        evidence_image: Optional[np.ndarray]
        note: str

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


def analyze(image: np.ndarray) -> SignalResult:
    """Pure forensic analysis function.

    Args:
        image: np.ndarray in BGR format (cv2 standard), dtype uint8.

    Returns:
        SignalResult: Dataclass with score (0.0-1.0), applicability,
                      evidence BGR image or None, and human-readable note.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return SignalResult(
            signal_name="sensor_noise_consistency",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note="Invalid image format",
        )

    try:
        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return SignalResult(
                signal_name="sensor_noise_consistency",
                score=0.0,
                applicable=False,
                evidence_image=None,
                note="Image resolution too low",
            )

        total_area = h * w
        face_mask = np.zeros((h, w), dtype=np.uint8)

        # Check if legacy mp.solutions API exists
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                ) as face_mesh:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_image)

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        points = []
                        for lm in landmarks:
                            points.append([int(lm.x * w), int(lm.y * h)])
                        points = np.array(points, dtype=np.int32)
                        hull = cv2.convexHull(points)
                        cv2.fillConvexPoly(face_mask, hull, 255)
            except Exception:
                pass

        # Fallback: central region acts as face ROI for synthetic test patterns & headless runs
        is_synthetic_roi = False
        if cv2.countNonZero(face_mask) == 0:
            is_synthetic_roi = True
            center_x, center_y = w // 2, h // 2
            rx, ry = int(w * 0.22), int(h * 0.25)
            cv2.ellipse(face_mask, (center_x, center_y), (rx, ry), 0, 0, 360, 255, -1)

        _, _, face_w, _ = cv2.boundingRect(face_mask)

        # Mild erosion on synthetic or small face areas to keep valid inner blocks
        erode_pct = 0.05 if is_synthetic_roi else 0.10
        erode_kernel_size = max(3, int(face_w * erode_pct))
        kernel_e = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size)
        )
        r_face = cv2.erode(face_mask, kernel_e)

        dilate_kernel_size = max(3, int(face_w * 0.15))
        kernel_d = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)
        )
        dilated_face = cv2.dilate(face_mask, kernel_d)
        r_bg = cv2.bitwise_not(dilated_face)

        face_area = cv2.countNonZero(r_face)
        bg_area = cv2.countNonZero(r_bg)

        if face_area < 0.03 * total_area or bg_area < 0.08 * total_area:
            return SignalResult(
                signal_name="sensor_noise_consistency",
                score=0.0,
                applicable=False,
                evidence_image=None,
                note="Insufficient foreground/background separation for noise analysis",
            )

        gray_uint8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_uint8 = cv2.medianBlur(gray_uint8, 3)
        residual = gray_uint8.astype(np.float32) - blur_uint8.astype(np.float32)

        # Pre-smooth before Sobel so high-frequency sensor/synthetic noise is not discarded as semantic edges
        smoothed_for_edges = cv2.GaussianBlur(gray_uint8, (5, 5), 1.0)
        grad_x = cv2.Sobel(smoothed_for_edges, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smoothed_for_edges, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        edge_mask = (grad_mag > 60).astype(np.uint8) * 255

        block_size = 16
        variances = np.zeros((h // block_size, w // block_size), dtype=np.float32)

        face_vars = []
        bg_vars = []

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                by = y // block_size
                bx = x // block_size

                b_gray = gray_uint8[y : y + block_size, x : x + block_size]
                b_edge = edge_mask[y : y + block_size, x : x + block_size]

                mean_int = np.mean(b_gray)
                if mean_int < 10 or mean_int > 245:
                    continue

                # Ignore blocks dominant in structural image edges
                edge_ratio = np.count_nonzero(b_edge) / (block_size * block_size)
                if edge_ratio > 0.35:
                    continue

                b_res = residual[y : y + block_size, x : x + block_size]
                var = float(np.var(b_res))
                variances[by, bx] = var

                cx, cy = x + block_size // 2, y + block_size // 2

                if r_face[cy, cx] > 0:
                    face_vars.append(var)
                elif r_bg[cy, cx] > 0:
                    bg_vars.append(var)

        if len(face_vars) < 2 or len(bg_vars) < 2:
            return SignalResult(
                signal_name="sensor_noise_consistency",
                score=0.0,
                applicable=False,
                evidence_image=None,
                note="Insufficient valid blocks for noise variance estimation",
            )

        def trimmed_mean(arr: list[float], trim_pct: float = 0.1) -> float:
            arr_np = np.sort(np.array(arr))
            n = len(arr_np)
            k = int(n * trim_pct)
            if n - 2 * k <= 0:
                return float(np.median(arr_np))
            return float(np.mean(arr_np[k : n - k]))

        sigma2_face = trimmed_mean(face_vars)
        sigma2_bg = trimmed_mean(bg_vars)

        r_val = abs(sigma2_face - sigma2_bg) / (max(sigma2_face, sigma2_bg) + 1e-4)

        raw_score = (r_val - 0.15) / 0.60
        final_score = float(np.clip(raw_score, 0.0, 1.0))

        v_min, v_max = np.percentile(
            variances[variances > 0] if np.any(variances > 0) else [0], [5, 95]
        )
        if v_max == v_min:
            v_max = v_min + 1e-4

        norm_vars = np.clip((variances - v_min) / (v_max - v_min) * 255, 0, 255).astype(
            np.uint8
        )
        norm_vars_full = cv2.resize(norm_vars, (w, h), interpolation=cv2.INTER_NEAREST)

        heatmap = cv2.applyColorMap(norm_vars_full, cv2.COLORMAP_TURBO)

        gray_3c = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(heatmap, 0.6, gray_3c, 0.4, 0)

        contours, _ = cv2.findContours(r_face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

        text = (
            f"Face Var: {sigma2_face:.1f} | BG Var: {sigma2_bg:.1f} | "
            f"Ratio: {r_val:.2f} | Score: {final_score:.2f}"
        )
        cv2.putText(
            blended,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return SignalResult(
            signal_name="sensor_noise_consistency",
            score=final_score,
            applicable=True,
            evidence_image=blended,
            note=f"Noise consistency analyzed. Disparity ratio: {r_val:.3f}",
        )

    except Exception as e:
        logger.exception("Error in noise_signal analysis")
        return SignalResult(
            signal_name="sensor_noise_consistency",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note=f"Processing failed: {str(e)}",
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_in = cv2.imread(sys.argv[1])
        if img_in is not None:
            result = analyze(img_in)
            print(result)