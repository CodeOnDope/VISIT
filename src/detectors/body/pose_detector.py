"""
Pose detector for the VISIT Museum Tracker system.

Detects human body pose keypoints using MediaPipe Pose or similar.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("PoseDetector")

class PoseDetector(DetectorBase):
    """Detects body pose keypoints using MediaPipe Pose."""

    def __init__(self, config=None):
        """
        Initialize PoseDetector.

        Args:
            config (dict, optional): Configuration options including:
                - min_detection_confidence (float): Minimum detection confidence
                - min_tracking_confidence (float): Minimum tracking confidence
        """
        super().__init__(config)
        self.config = config or {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5)
        )

        self.results = None

    def process_frame(self, frame):
        """
        Process a frame to detect pose landmarks.

        Args:
            frame (np.ndarray): BGR image frame.

        Returns:
            dict: Pose landmarks and annotated frame.
        """
        if not self.is_running:
            return {"pose_landmarks": None, "annotated_frame": frame.copy()}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb_frame)
        annotated_frame = frame.copy()

        if self.results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            landmarks = []
            ih, iw, _ = frame.shape
            for lm in self.results.pose_landmarks.landmark:
                landmarks.append((int(lm.x * iw), int(lm.y * ih)))
        else:
            landmarks = None

        return {"pose_landmarks": landmarks, "annotated_frame": annotated_frame}

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Pose Detector",
            "description": "Detects body pose keypoints using MediaPipe Pose",
            "config_options": {
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
 
