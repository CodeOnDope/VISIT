"""
Finger detector for the VISIT Museum Tracker system.

Detects individual finger positions and states using hand landmarks.
"""

import logging
import numpy as np

from src.core.detector_base import DetectorBase

logger = logging.getLogger("FingerDetector")

class FingerDetector(DetectorBase):
    """Detects finger positions and states based on hand landmarks."""

    FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tip landmark indices

    def __init__(self, config=None):
        """
        Initialize FingerDetector.

        Args:
            config (dict, optional): Configuration options.
        """
        super().__init__(config)
        self.config = config or {}

    def process_frame(self, hand_landmarks):
        """
        Detect finger states from hand landmarks.

        Args:
            hand_landmarks (list): List of (x, y) tuples representing hand landmarks.

        Returns:
            dict: Finger states, e.g. which fingers are extended.
        """
        if not self.is_running:
            return {"finger_states": {}}

        if not hand_landmarks or len(hand_landmarks) < 21:
            return {"finger_states": {}}

        finger_states = {}
        try:
            # Example heuristic: check if tip is above the PIP joint in y-axis (for index finger as sample)
            # This is a simple heuristic; replace with your actual finger detection logic

            for i, tip_idx in enumerate(self.FINGER_TIPS):
                tip_y = hand_landmarks[tip_idx][1]
                pip_y = hand_landmarks[tip_idx - 2][1]  # PIP joint is usually 2 indices before tip

                finger_states[f"finger_{i}"] = tip_y < pip_y  # True if finger extended (tip above pip)

        except Exception as e:
            logger.error(f"Error detecting finger states: {e}")

        return {"finger_states": finger_states}

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Finger Detector",
            "description": "Detects finger states based on hand landmarks",
            "config_options": {}
        }
 
