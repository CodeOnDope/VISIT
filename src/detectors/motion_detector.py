"""
Motion detector module for the VISIT Museum Tracker system.

Detects motion areas in video frames using frame differencing and thresholding.
"""

import cv2
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("MotionDetector")

class MotionDetector(DetectorBase):
    """Detects motion in video frames."""

    def __init__(self, config=None):
        """
        Initialize MotionDetector.

        Args:
            config (dict, optional): Configuration options including:
                - threshold (int): Threshold value for motion detection (default 25)
                - blur_size (int): Gaussian blur kernel size (default 5)
                - min_area (int): Minimum area size to consider motion (default 500)
        """
        super().__init__(config)
        self.config = config or {
            "threshold": 25,
            "blur_size": 5,
            "min_area": 500,
            "show_motion": True,
            "show_threshold": False
        }

        self.previous_frame = None

    def process_frame(self, frame):
        """
        Detect motion regions in the frame.

        Args:
            frame (np.ndarray): BGR video frame.

        Returns:
            dict: Motion detection results including motion regions and annotated frame.
        """
        if not self.is_running:
            return {"motion_detected": False, "motion_regions": [], "annotated_frame": frame.copy()}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_size = self.config.get("blur_size", 5)
        if blur_size % 2 == 0:
            blur_size += 1
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        motion_detected = False
        motion_regions = []
        annotated_frame = frame.copy()

        if self.previous_frame is None:
            self.previous_frame = gray
            return {"motion_detected": False, "motion_regions": [], "annotated_frame": annotated_frame}

        frame_diff = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(frame_diff, self.config.get("threshold", 25), 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.config.get("min_area", 500):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            motion_regions.append((x, y, w, h))
            motion_detected = True
            if self.config.get("show_motion", True):
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.config.get("show_threshold", False):
            # Display threshold image in a corner
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            h, w = thresh_color.shape[:2]
            annotated_frame[0:h, 0:w] = thresh_color

        self.previous_frame = gray

        return {
            "motion_detected": motion_detected,
            "motion_regions": motion_regions,
            "annotated_frame": annotated_frame
        }

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Motion Detector",
            "description": "Detects motion regions using frame differencing",
            "config_options": {
                "threshold": {
                    "type": "int",
                    "default": 25,
                    "description": "Threshold for frame differencing"
                },
                "blur_size": {
                    "type": "int",
                    "default": 5,
                    "description": "Gaussian blur kernel size (odd number)"
                },
                "min_area": {
                    "type": "int",
                    "default": 500,
                    "description": "Minimum contour area to consider as motion"
                },
                "show_motion": {
                    "type": "bool",
                    "default": True,
                    "description": "Show motion bounding boxes"
                },
                "show_threshold": {
                    "type": "bool",
                    "default": False,
                    "description": "Show threshold image overlay"
                }
            }
        }
