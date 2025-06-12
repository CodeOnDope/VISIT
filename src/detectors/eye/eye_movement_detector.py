"""
Eye movement detector for the VISIT Museum Tracker system.

Analyzes eye landmarks to estimate gaze direction and movement.
"""

import cv2
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("EyeMovementDetector")

class EyeMovementDetector(DetectorBase):
    """Detects and estimates eye gaze direction."""

    def __init__(self, config=None):
        """
        Initialize EyeMovementDetector.

        Args:
            config (dict, optional): Configuration options including:
                - movement_sensitivity (float): Sensitivity threshold for movement detection (default 0.1)
        """
        super().__init__(config)
        self.config = config or {
            "movement_sensitivity": 0.1
        }

        self.prev_left_eye_center = None
        self.prev_right_eye_center = None

    def process_frame(self, landmarks):
        """
        Analyze eye landmarks to estimate movement direction.

        Args:
            landmarks (dict): Contains eye landmarks points with keys 'left_eye' and 'right_eye'.

        Returns:
            dict: Movement results containing gaze direction and annotated frame info.
        """
        if not self.is_running:
            return {"gaze_direction": None}

        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return {"gaze_direction": None}

        left_center = self._calculate_eye_center(left_eye)
        right_center = self._calculate_eye_center(right_eye)

        gaze_direction = None
        if self.prev_left_eye_center and self.prev_right_eye_center:
            left_movement = np.array(left_center) - np.array(self.prev_left_eye_center)
            right_movement = np.array(right_center) - np.array(self.prev_right_eye_center)

            avg_movement = (left_movement + right_movement) / 2.0

            if np.linalg.norm(avg_movement) > self.config["movement_sensitivity"]:
                if abs(avg_movement[0]) > abs(avg_movement[1]):
                    if avg_movement[0] > 0:
                        gaze_direction = "Right"
                    else:
                        gaze_direction = "Left"
                else:
                    if avg_movement[1] > 0:
                        gaze_direction = "Down"
                    else:
                        gaze_direction = "Up"

        self.prev_left_eye_center = left_center
        self.prev_right_eye_center = right_center

        logger.debug(f"Gaze direction: {gaze_direction}")

        return {"gaze_direction": gaze_direction}

    def _calculate_eye_center(self, eye_points):
        """
        Calculate the center of an eye given its landmark points.

        Args:
            eye_points (list): List of (x, y) tuples.

        Returns:
            tuple: (x, y) center point.
        """
        x_coords = [p[0] for p in eye_points]
        y_coords = [p[1] for p in eye_points]
        center_x = int(sum(x_coords) / len(x_coords))
        center_y = int(sum(y_coords) / len(y_coords))
        return (center_x, center_y)

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Eye Movement Detector",
            "description": "Estimates eye gaze direction from landmarks",
            "config_options": {
                "movement_sensitivity": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.1,
                    "description": "Threshold for gaze movement detection"
                }
            }
        }
