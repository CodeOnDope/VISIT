"""
Face detector using MediaPipe for the VISIT Museum Tracker system.

This module implements face detection using MediaPipe's FaceDetection solution.
It extends the base Detector class and provides methods for processing frames,
drawing detections, and configuration management.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("FaceDetector")

class FaceDetector(DetectorBase):
    """MediaPipe-based face detector."""

    def __init__(self, config=None):
        """
        Initialize the FaceDetector.

        Args:
            config (dict, optional): Configuration options including:
                - min_detection_confidence (float): Minimum confidence threshold (default 0.5)
                - model_selection (int): 0 for short range, 1 for full range (default 0)
        """
        super().__init__(config)
        self.config = config or {
            "min_detection_confidence": 0.5,
            "model_selection": 0
        }

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
            model_selection=self.config.get("model_selection", 0)
        )

        self.results = []

    def process_frame(self, frame):
        """
        Detect faces in a video frame.

        Args:
            frame (np.ndarray): Input BGR frame from video capture.

        Returns:
            dict: Detection results containing 'faces' list and 'annotated_frame'.
        """
        if not self.is_running:
            return {"faces": [], "annotated_frame": frame.copy()}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.detector.process(rgb_frame)
        annotated_frame = frame.copy()
        faces = []

        if mp_results.detections:
            ih, iw, _ = frame.shape
            for detection in mp_results.detections:
                bbox = detection.location_data.relative_bounding_box
                xmin = max(0, int(bbox.xmin * iw))
                ymin = max(0, int(bbox.ymin * ih))
                width = int(bbox.width * iw)
                height = int(bbox.height * ih)
                xmax = min(iw, xmin + width)
                ymax = min(ih, ymin + height)

                landmarks = {}
                for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                    px = min(int(landmark.x * iw), iw - 1)
                    py = min(int(landmark.y * ih), ih - 1)
                    landmarks[f"landmark_{idx}"] = (px, py)

                face_data = {
                    "bbox": (xmin, ymin, xmax, ymax),
                    "bbox_confidence": detection.score[0] if detection.score else 0.0,
                    "landmarks": landmarks
                }
                faces.append(face_data)
                self._draw_detection(annotated_frame, face_data)

        self.results = {
            "faces": faces,
            "annotated_frame": annotated_frame
        }
        return self.results

    def _draw_detection(self, image, face_data):
        """
        Draw bounding box and landmarks on the image.

        Args:
            image (np.ndarray): Image to draw on.
            face_data (dict): Detected face info with bbox and landmarks.
        """
        xmin, ymin, xmax, ymax = face_data["bbox"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        confidence = face_data["bbox_confidence"]
        label = f"Face: {confidence:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for _, (px, py) in face_data["landmarks"].items():
            cv2.circle(image, (px, py), 2, (0, 0, 255), 2)

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Face Detector",
            "description": "Detects faces using MediaPipe FaceDetection",
            "config_options": {
                "min_detection_confidence": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "description": "Minimum detection confidence threshold"
                },
                "model_selection": {
                    "type": "int",
                    "options": [0, 1],
                    "default": 0,
                    "description": "0=Short range, 1=Full range detection model"
                }
            }
        }
