 
"""
Eye detector module for the VISIT Museum Tracker system.

Detects eyes and optionally tracks blink and gaze using MediaPipe or custom methods.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("EyeDetector")

class EyeDetector(DetectorBase):
    """Detects eyes and analyzes blink/gaze."""

    def __init__(self, config=None):
        """
        Initialize EyeDetector.

        Args:
            config (dict, optional): Configuration options including:
                - min_detection_confidence (float): Confidence threshold for detection
                - min_tracking_confidence (float): Confidence threshold for tracking
        """
        super().__init__(config)
        self.config = config or {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5),
        )

        self.results = []

    def process_frame(self, frame):
        """
        Detect eyes and analyze gaze/blink in the frame.

        Args:
            frame (np.ndarray): BGR image frame.

        Returns:
            dict: Contains detected eye landmarks and annotated frame.
        """
        if not self.is_running:
            return {"eye_landmarks": [], "annotated_frame": frame.copy()}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        annotated_frame = frame.copy()
        eye_landmarks = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                left_eye_points = []
                right_eye_points = []

                # MediaPipe face mesh eye landmark indices (example indices)
                left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
                right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]

                for idx in left_eye_indices:
                    lm = face_landmarks.landmark[idx]
                    left_eye_points.append((int(lm.x * iw), int(lm.y * ih)))

                for idx in right_eye_indices:
                    lm = face_landmarks.landmark[idx]
                    right_eye_points.append((int(lm.x * iw), int(lm.y * ih)))

                eye_landmarks.append({
                    "left_eye": left_eye_points,
                    "right_eye": right_eye_points
                })

                self._draw_eye_landmarks(annotated_frame, left_eye_points, right_eye_points)

        self.results = {
            "eye_landmarks": eye_landmarks,
            "annotated_frame": annotated_frame
        }
        return self.results

    def _draw_eye_landmarks(self, image, left_eye_points, right_eye_points):
        """
        Draw eye landmarks on the image.

        Args:
            image (np.ndarray): Image to draw on.
            left_eye_points (list): List of (x, y) points for left eye.
            right_eye_points (list): List of (x, y) points for right eye.
        """
        for point in left_eye_points:
            cv2.circle(image, point, 2, (0, 255, 0), -1)
        for point in right_eye_points:
            cv2.circle(image, point, 2, (0, 255, 0), -1)

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Eye Detector",
            "description": "Detects eyes and tracks blink/gaze using MediaPipe Face Mesh",
            "config_options": {
                "min_detection_confidence": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "description": "Minimum detection confidence threshold"
                },
                "min_tracking_confidence": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "description": "Minimum tracking confidence threshold"
                }
            }
        }
