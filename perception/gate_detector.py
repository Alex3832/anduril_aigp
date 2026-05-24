"""OpenCV-based gate detection from raw camera frames."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config


@dataclass
class GateObservation:
    """Pixel-space observation of a detected gate."""

    center_px: Tuple[float, float]   # (u, v) in image coordinates
    area_px: float                    # bounding-box area (pixels²)
    confidence: float                 # 0–1 detection confidence


class GateDetector:
    """Detect race gates in camera images using HSV colour segmentation.

    Workflow:
        1. Convert BGR frame to HSV.
        2. Threshold on gate colour to build a binary mask.
        3. Find contours and filter by area.
        4. Return the largest qualifying contour as a GateObservation.
    """

    def __init__(self, hsv_lower: Tuple[int, int, int],
                 hsv_upper: Tuple[int, int, int],
                 min_area_px: int) -> None:
        """
        Args:
            hsv_lower:   Lower HSV bound for gate colour (H, S, V).
            hsv_upper:   Upper HSV bound for gate colour.
            min_area_px: Minimum contour area to consider a valid detection.
        """
        self._lower = np.array(hsv_lower, dtype=np.uint8)
        self._upper = np.array(hsv_upper, dtype=np.uint8)
        self._min_area = min_area_px

    @classmethod
    def from_config(cls) -> "GateDetector":
        """Construct using parameters from config.py."""
        return cls(
            hsv_lower=config.GATE_HSV_LOWER,
            hsv_upper=config.GATE_HSV_UPPER,
            min_area_px=config.MIN_GATE_AREA_PX,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[GateObservation]:
        """Run detection on a single BGR frame.

        Args:
            frame_bgr: H×W×3 uint8 image in BGR colour order.

        Returns:
            GateObservation for the best candidate, or None if nothing found.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self._min_area:
            return None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        confidence = min(area / (self._min_area * 10), 1.0)

        return GateObservation(center_px=(cx, cy), area_px=area, confidence=confidence)
