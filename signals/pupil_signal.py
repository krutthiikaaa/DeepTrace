"""
Workstream B: Biometric & Sensor-Noise Forensics
Pupil / Corneal Specular Reflection Mismatch Signal
"""
from __future__ import annotations

import logging
import math
import sys
from typing import Optional, Tuple

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


def _get_glint_and_centroid(
    crop: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], np.ndarray, float]]:
    """Extracts specular glint centroid and color from an iris crop."""
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    pct_98 = np.percentile(gray, 98)
    threshold_val = max(215.0, float(pct_98))

    _, thresh = cv2.threshold(gray, int(threshold_val), 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    if num_labels <= 1:
        return None

    largest_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i

    if max_area < 1:
        return None

    mask = (labels == largest_label).astype(np.uint8)
    mean_val = cv2.mean(gray, mask=mask)[0]

    if mean_val < 180:
        return None

    M = cv2.moments(thresh * mask)
    if M["m00"] == 0:
        return None

    gx = M["m10"] / M["m00"]
    gy = M["m01"] / M["m00"]

    mean_color = cv2.mean(crop, mask=mask)[:3]
    return (gx, gy), np.array(mean_color), float(max_area)


def analyze(image: np.ndarray) -> SignalResult:
    """Pure forensic analysis function.

    Args:
        image: np.ndarray in BGR format (cv2 standard), dtype uint8.

    Returns:
        SignalResult: Dataclass with score (0.0-1.0), applicability,
                      evidence BGR image or None, and human-readable note.
    """
    if mp is None:
        return SignalResult(
            signal_name="pupil_specular_mismatch",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note="MediaPipe not installed",
        )

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        return SignalResult(
            signal_name="pupil_specular_mismatch",
            score=0.0,
            applicable=False,
            evidence_image=None,
            note="Invalid image format",
        )

    try:
        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return SignalResult(
                signal_name="pupil_specular_mismatch",
                score=0.0,
                applicable=False,
                evidence_image=None,
                note="Image resolution too low",
            )

        # Check if legacy MediaPipe solutions API exists
        has_mesh = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        if not has_mesh:
            return SignalResult(
                signal_name="pupil_specular_mismatch",
                score=0.0,
                applicable=False,
                evidence_image=None,
                note="Face or eye landmarks not detected",
            )

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return SignalResult(
                    signal_name="pupil_specular_mismatch",
                    score=0.0,
                    applicable=False,
                    evidence_image=None,
                    note="Face or eye landmarks not detected",
                )

            landmarks = results.multi_face_landmarks[0].landmark

            def get_px(idx: int) -> Tuple[float, float]:
                return (landmarks[idx].x * w, landmarks[idx].y * h)

            l_iris_c = get_px(468)
            r_iris_c = get_px(473)

            l_eye_inner = get_px(133)
            r_eye_inner = get_px(362)

            iod = math.hypot(
                l_eye_inner[0] - r_eye_inner[0], l_eye_inner[1] - r_eye_inner[1]
            )
            if iod < 30:
                return SignalResult(
                    signal_name="pupil_specular_mismatch",
                    score=0.0,
                    applicable=False,
                    evidence_image=None,
                    note="Face off-angle or eye resolution insufficient for specular analysis",
                )

            face_center_x = (l_eye_inner[0] + r_eye_inner[0]) / 2.0
            nose_tip = get_px(1)
            yaw_ratio = abs(nose_tip[0] - face_center_x) / (iod + 1e-6)
            if yaw_ratio > 1.0:
                return SignalResult(
                    signal_name="pupil_specular_mismatch",
                    score=0.0,
                    applicable=False,
                    evidence_image=None,
                    note="Face off-angle or eye resolution insufficient for specular analysis",
                )

            l_iris_r = math.hypot(
                get_px(469)[0] - l_iris_c[0], get_px(469)[1] - l_iris_c[1]
            )
            r_iris_r = math.hypot(
                get_px(474)[0] - r_iris_c[0], get_px(474)[1] - r_iris_c[1]
            )

            if l_iris_r < 4 or r_iris_r < 4:
                return SignalResult(
                    signal_name="pupil_specular_mismatch",
                    score=0.0,
                    applicable=False,
                    evidence_image=None,
                    note="Iris resolution too low",
                )

            def crop_iris(
                center: Tuple[float, float], radius: float
            ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
                margin = int(radius * 1.5)
                cx, cy = int(center[0]), int(center[1])
                x1 = max(0, cx - margin)
                y1 = max(0, cy - margin)
                x2 = min(w, cx + margin)
                y2 = min(h, cy + margin)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    return None, (0, 0)
                return image[y1:y2, x1:x2].copy(), (x1, y1)

            l_crop, l_offset = crop_iris(l_iris_c, l_iris_r)
            r_crop, r_offset = crop_iris(r_iris_c, r_iris_r)

            if l_crop is None or r_crop is None:
                return SignalResult(
                    signal_name="pupil_specular_mismatch",
                    score=0.0,
                    applicable=False,
                    evidence_image=None,
                    note="Iris resolution too low",
                )

            l_res = _get_glint_and_centroid(l_crop)
            r_res = _get_glint_and_centroid(r_crop)

            if l_res is None or r_res is None:
                return SignalResult(
                    signal_name="pupil_specular_mismatch",
                    score=0.0,
                    applicable=False,
                    evidence_image=None,
                    note="No valid specular corneal reflections detected in one or both eyes",
                )

            (l_gx, l_gy), l_color, _ = l_res
            (r_gx, r_gy), r_color, _ = r_res

            l_glint_global = (l_gx + l_offset[0], l_gy + l_offset[1])
            r_glint_global = (r_gx + r_offset[0], r_gy + r_offset[1])

            v_l = (
                np.array(
                    [
                        l_glint_global[0] - l_iris_c[0],
                        l_glint_global[1] - l_iris_c[1],
                    ]
                )
                / l_iris_r
            )
            v_r = (
                np.array(
                    [
                        r_glint_global[0] - r_iris_c[0],
                        r_glint_global[1] - r_iris_c[1],
                    ]
                )
                / r_iris_r
            )

            d_pos = float(np.linalg.norm(v_l - v_r))
            d_color = float(
                np.linalg.norm(l_color - r_color) / (255.0 * math.sqrt(3))
            )

            raw_score = 0.7 * min(1.0, d_pos / 0.4) + 0.3 * min(1.0, d_color / 0.3)
            final_score = float(np.clip(raw_score, 0.0, 1.0))

            out_size = 150
            l_viz = cv2.resize(
                l_crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC
            )
            r_viz = cv2.resize(
                r_crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC
            )

            def draw_markers(
                viz: np.ndarray,
                iris_c: Tuple[float, float],
                glint_c: Tuple[float, float],
                offset: Tuple[int, int],
                crop_shape: Tuple[int, ...],
            ) -> np.ndarray:
                sc_x = out_size / crop_shape[1]
                sc_y = out_size / crop_shape[0]
                ic_x = (iris_c[0] - offset[0]) * sc_x
                ic_y = (iris_c[1] - offset[1]) * sc_y
                gc_x = (glint_c[0] - offset[0]) * sc_x
                gc_y = (glint_c[1] - offset[1]) * sc_y
                cv2.drawMarker(
                    viz,
                    (int(ic_x), int(ic_y)),
                    (255, 255, 0),
                    cv2.MARKER_CROSS,
                    10,
                    2,
                )
                cv2.circle(viz, (int(gc_x), int(gc_y)), 5, (0, 0, 255), 2)
                return viz

            l_viz = draw_markers(
                l_viz, l_iris_c, l_glint_global, l_offset, l_crop.shape
            )
            r_viz = draw_markers(
                r_viz, r_iris_c, r_glint_global, r_offset, r_crop.shape
            )

            composite = np.hstack((l_viz, r_viz))

            banner_h = 30
            banner = np.zeros((banner_h, out_size * 2, 3), dtype=np.uint8)
            text = f"Disparity: {d_pos:.3f} | Score: {final_score:.2f}"
            cv2.putText(
                banner,
                text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            evidence = np.vstack((composite, banner))

            return SignalResult(
                signal_name="pupil_specular_mismatch",
                score=final_score,
                applicable=True,
                evidence_image=evidence,
                note=f"Specularity compared successfully. Mismatch score: {final_score:.2f}",
            )

    except Exception as e:
        logger.exception("Error in pupil_signal analysis")
        return SignalResult(
            signal_name="pupil_specular_mismatch",
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