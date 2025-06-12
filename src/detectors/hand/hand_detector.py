 
"""
Hand detector module for the VISIT Museum Tracker system.

Detects and tracks hands in video frames using MediaPipe Hands solution.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("HandDetector")

class HandDetector(DetectorBase):
    """Detects and tracks hands using MediaPipe Hands."""
    
    
    
    def __init__(self, config=None):
        """
        Initialize the HandDetector.

        Args:
            config (dict, optional): Configuration including:
                - max_num_hands (int): Max hands to detect (default 2)
                - min_detection_confidence (float): Minimum confidence for detection (default 0.5)
                - min_tracking_confidence (float): Minimum confidence for tracking (default 0.5)
        """
        super().__init__(config)
        self.config = config or {
            "max_num_hands": 2,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        }

        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            max_num_hands=self.config.get("max_num_hands", 2),
            min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5)
        )

        self.results = []

    def process_frame(self, frame):
        """
        Detect hands and landmarks in a video frame.

        Args:
            frame (np.ndarray): BGR image frame.

        Returns:
            dict: Contains list of detected hands and annotated frame.
        """
        if not self.is_running:
            return {"hands": [], "annotated_frame": frame.copy()}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.detector.process(rgb_frame)
        annotated_frame = frame.copy()
        hands = []

        if self.results.multi_hand_landmarks:
            ih, iw, _ = frame.shape
            for hand_landmarks in self.results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    landmarks.append((x, y))

                hands.append(landmarks)
                self._draw_hand_landmarks(annotated_frame, hand_landmarks)

        return {"hands": hands, "annotated_frame": annotated_frame}
    def process(self, frame):
        """
        Process the frame and return list of bounding boxes for hands detected.
        This method will be used by the detection pipeline.
        """
        if not self.is_running:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.detector.process(rgb_frame)

        hand_bboxes = []
        h, w, _ = frame.shape
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                xmin = int(min(x_coords) * w)
                xmax = int(max(x_coords) * w)
                ymin = int(min(y_coords) * h)
                ymax = int(max(y_coords) * h)

                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                hand_bboxes.append(bbox)

        return hand_bboxes
    
    def _draw_hand_landmarks(self, image, hand_landmarks):
        """
        Draw hand landmarks on the image.

        Args:
            image (np.ndarray): Image to draw on.
            hand_landmarks: MediaPipe hand landmarks object.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 128, 255), thickness=2)
        )

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Hand Detector",
            "description": "Detects and tracks hands using MediaPipe Hands",
            "config_options": {
                "max_num_hands": {
                    "type": "int",
                    "default": 2,
                    "description": "Maximum number of hands to detect"
                },
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
