"""
Eye blink detector for the VISIT Museum Tracker system.

Detects blinks using eye aspect ratio or landmarks over video frames.
"""

import cv2
import numpy as np
import time
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("EyeBlinkDetector")

class EyeBlinkDetector(DetectorBase):
    """Detects eye blinks based on landmarks and timing."""

    def __init__(self, config=None):
        """
        Initialize EyeBlinkDetector.

        Args:
            config (dict, optional): Configuration options including:
                - blink_threshold (float): EAR threshold to detect blink (default 0.2)
                - consecutive_frames (int): Number of consecutive frames below threshold to count blink (default 3)
        """
        super().__init__(config)
        self.config = config or {
            "blink_threshold": 0.2,
            "consecutive_frames": 3,
        }

        self.blink_counter = 0
        self.total_blinks = 0
        self.eye_closed = False

    def process_frame(self, landmarks):
        """
        Process eye landmarks to detect blinks.

        Args:
            landmarks (dict): Contains eye landmarks points, typically from eye detector.

        Returns:
            dict: Contains blink count and status.
        """
        if not self.is_running:
            return {"blink_count": self.total_blinks, "blink_detected": False}

        blink_detected = False
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return {"blink_count": self.total_blinks, "blink_detected": False}

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < self.config["blink_threshold"]:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.config["consecutive_frames"]:
                self.total_blinks += 1
                blink_detected = True
                logger.debug(f"Blink detected. Total blinks: {self.total_blinks}")
            self.blink_counter = 0

        return {"blink_count": self.total_blinks, "blink_detected": blink_detected}

    def _eye_aspect_ratio(self, eye_points):
        """
        Compute Eye Aspect Ratio (EAR).

        Args:
            eye_points (list): List of (x, y) tuples for eye landmarks.

        Returns:
            float: EAR value.
        """
        # Calculate the euclidean distances between vertical eye landmarks
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # Calculate the euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        ear = (A + B) / (2.0 * C)
        return ear

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Eye Blink Detector",
            "description": "Detects eye blinks using Eye Aspect Ratio (EAR)",
            "config_options": {
                "blink_threshold": {
                    "type": "float",
                    "min": 0.1,
                    "max": 0.3,
                    "default": 0.2,
                    "description": "EAR threshold below which eye is considered closed"
                },
                "consecutive_frames": {
                    "type": "int",
                    "default": 3,
                    "description": "Number of consecutive frames EAR must be below threshold"
                }
            }
        }
"""
Eye blink detector for the VISIT Museum Tracker system.

Detects blinks using eye aspect ratio or landmarks over video frames.
"""

import cv2
import numpy as np
import time
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("EyeBlinkDetector")

class EyeBlinkDetector(DetectorBase):
    """Detects eye blinks based on landmarks and timing."""

    def __init__(self, config=None):
        """
        Initialize EyeBlinkDetector.

        Args:
            config (dict, optional): Configuration options including:
                - blink_threshold (float): EAR threshold to detect blink (default 0.2)
                - consecutive_frames (int): Number of consecutive frames below threshold to count blink (default 3)
        """
        super().__init__(config)
        self.config = config or {
            "blink_threshold": 0.2,
            "consecutive_frames": 3,
        }

        self.blink_counter = 0
        self.total_blinks = 0
        self.eye_closed = False

    def process_frame(self, landmarks):
        """
        Process eye landmarks to detect blinks.

        Args:
            landmarks (dict): Contains eye landmarks points, typically from eye detector.

        Returns:
            dict: Contains blink count and status.
        """
        if not self.is_running:
            return {"blink_count": self.total_blinks, "blink_detected": False}

        blink_detected = False
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])

        if not left_eye or not right_eye:
            return {"blink_count": self.total_blinks, "blink_detected": False}

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < self.config["blink_threshold"]:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.config["consecutive_frames"]:
                self.total_blinks += 1
                blink_detected = True
                logger.debug(f"Blink detected. Total blinks: {self.total_blinks}")
            self.blink_counter = 0

        return {"blink_count": self.total_blinks, "blink_detected": blink_detected}

    def _eye_aspect_ratio(self, eye_points):
        """
        Compute Eye Aspect Ratio (EAR).

        Args:
            eye_points (list): List of (x, y) tuples for eye landmarks.

        Returns:
            float: EAR value.
        """
        # Calculate the euclidean distances between vertical eye landmarks
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        # Calculate the euclidean distance between horizontal eye landmarks
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        ear = (A + B) / (2.0 * C)
        return ear

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Eye Blink Detector",
            "description": "Detects eye blinks using Eye Aspect Ratio (EAR)",
            "config_options": {
                "blink_threshold": {
                    "type": "float",
                    "min": 0.1,
                    "max": 0.3,
                    "default": 0.2,
                    "description": "EAR threshold below which eye is considered closed"
                },
                "consecutive_frames": {
                    "type": "int",
                    "default": 3,
                    "description": "Number of consecutive frames EAR must be below threshold"
                }
            }
        }
 
