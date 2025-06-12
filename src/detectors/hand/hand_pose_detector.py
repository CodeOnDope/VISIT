
"""
Hand pose detector for the VISIT Museum Tracker system.

Classifies hand poses based on detected landmarks and gestures.
"""

import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("HandPoseDetector")

class HandPoseDetector(DetectorBase):
    """Classifies hand poses from landmarks and gestures."""

    def __init__(self, config=None):
        """
        Initialize HandPoseDetector.

        Args:
            config (dict, optional): Configuration options including:
                - pose_classes (list): List of known poses to classify
        """
        super().__init__(config)
        self.config = config or {
            "pose_classes": ["open_hand", "fist", "peace", "thumbs_up"]
        }

    def process_frame(self, gesture_result):
        """
        Classify hand pose from gesture detection result.

        Args:
            gesture_result (dict): Output from gesture detector, e.g. {"gesture": "fist"}

        Returns:
            dict: Hand pose classification with pose name and confidence.
        """
        if not self.is_running:
            return {"pose": None, "confidence": 0.0}

        gesture = gesture_result.get("gesture")
        confidence = gesture_result.get("confidence", 0.0)

        pose = None
        if gesture in self.config.get("pose_classes", []):
            pose = gesture

        return {"pose": pose, "confidence": confidence}

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Hand Pose Detector",
            "description": "Classifies hand poses based on gestures",
            "config_options": {
                "pose_classes": {
                    "type": "list",
                    "default": ["open_hand", "fist", "peace", "thumbs_up"],
                    "description": "Known hand pose classes"
                }
            }
        }
