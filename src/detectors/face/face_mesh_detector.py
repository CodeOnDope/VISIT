 
"""
Face mesh detector using MediaPipe for detailed facial landmark detection.

This module detects a dense mesh of facial landmarks, enabling advanced
analysis such as gaze tracking, expression recognition, and more.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("FaceMeshDetector")

class FaceMeshDetector(DetectorBase):
    """MediaPipe Face Mesh detector."""

    def __init__(self, config=None):
        """
        Initialize the FaceMeshDetector.

        Args:
            config (dict, optional): Configuration including:
                - max_num_faces (int): Max faces to detect (default 1)
                - refine_landmarks (bool): Refine landmarks for iris (default False)
                - min_detection_confidence (float): Minimum detection confidence (default 0.5)
                - min_tracking_confidence (float): Minimum tracking confidence (default 0.5)
        """
        super().__init__(config)
        self.config = config or {
            "max_num_faces": 1,
            "refine_landmarks": False,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        }

        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config.get("max_num_faces", 1),
            refine_landmarks=self.config.get("refine_landmarks", False),
            min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5)
        )

        self.results = []

    def process_frame(self, frame):
        """
        Process a frame to detect face mesh landmarks.

        Args:
            frame (np.ndarray): Input BGR image frame.

        Returns:
            dict: Contains 'landmarks' list and 'annotated_frame'.
        """
        if not self.is_running:
            return {"landmarks": [], "annotated_frame": frame.copy()}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        annotated_frame = frame.copy()
        face_landmarks = []

        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                landmarks = []
                ih, iw, _ = frame.shape
                for lm in face_landmark.landmark:
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks.append((x, y))

                face_landmarks.append(landmarks)
                self._draw_landmarks(annotated_frame, face_landmark)

        self.results = {
            "landmarks": face_landmarks,
            "annotated_frame": annotated_frame
        }
        return self.results

    def _draw_landmarks(self, image, face_landmark):
        """
        Draw face mesh landmarks on the image.

        Args:
            image (np.ndarray): Image to draw on.
            face_landmark: MediaPipe face landmarks object.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            face_landmark,
            self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1)
        )

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Face Mesh Detector",
            "description": "Detects dense facial landmarks using MediaPipe Face Mesh",
            "config_options": {
                "max_num_faces": {
                    "type": "int",
                    "default": 1,
                    "description": "Maximum number of faces to detect"
                },
                "refine_landmarks": {
                    "type": "bool",
                    "default": False,
                    "description": "Refine landmarks for iris tracking"
                },
                "min_detection_confidence": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "description": "Minimum detection confidence"
                },
                "min_tracking_confidence": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "description": "Minimum tracking confidence"
                }
            }
        }
