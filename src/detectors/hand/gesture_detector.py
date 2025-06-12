"""
Gesture detector for the VISIT Museum Tracker system.

Recognizes hand gestures using detected finger states and landmarks.
"""

import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("GestureDetector")

class GestureDetector(DetectorBase):
    """Detects hand gestures based on finger positions and patterns."""

    def __init__(self, config=None):
        """
        Initialize GestureDetector.

        Args:
            config (dict, optional): Configuration options including gesture definitions.
        """
        super().__init__(config)
        self.config = config or {
            "gestures": {
                "fist": [False, False, False, False, False],
                "open_hand": [True, True, True, True, True],
                "peace": [False, True, True, False, False]
                # Add more gesture patterns here
            }
        }

    def process_frame(self, finger_states):
        """
        Detect gestures based on finger states.

        Args:
            finger_states (dict): Dict of finger extended states, e.g. {"finger_0": True, ...}

        Returns:
            dict: Detected gesture name and confidence.
        """
        if not self.is_running:
            return {"gesture": None, "confidence": 0.0}

        if not finger_states:
            return {"gesture": None, "confidence": 0.0}

        detected_gesture = None
        confidence = 0.0

        try:
            for gesture_name, pattern in self.config.get("gestures", {}).items():
                if self._match_pattern(finger_states, pattern):
                    detected_gesture = gesture_name
                    confidence = 1.0  # For now, simple binary confidence
                    break

        except Exception as e:
            logger.error(f"Error detecting gesture: {e}")

        return {"gesture": detected_gesture, "confidence": confidence}

    def _match_pattern(self, finger_states, pattern):
        """
        Match the finger states against a gesture pattern.

        Args:
            finger_states (dict): e.g. {"finger_0": True, "finger_1": False, ...}
            pattern (list): List of bools for fingers [thumb, index, middle, ring, pinky]

        Returns:
            bool: True if pattern matches.
        """
        for i, expected in enumerate(pattern):
            if finger_states.get(f"finger_{i}", False) != expected:
                return False
        return True

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Gesture Detector",
            "description": "Recognizes hand gestures from finger states",
            "config_options": {
                "gestures": {
                    "type": "dict",
                    "default": {
                        "fist": [False, False, False, False, False],
                        "open_hand": [True, True, True, True, True],
                        "peace": [False, True, True, False, False]
                    },
                    "description": "Gesture patterns mapping finger states"
                }
            }
        }
