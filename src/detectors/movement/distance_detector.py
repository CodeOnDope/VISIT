"""
Distance detector for the VISIT Museum Tracker system.

Estimates distances between visitors or objects based on detection data.
"""

import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("DistanceDetector")

class DistanceDetector(DetectorBase):
    """Estimates distances between tracked objects or visitors."""

    def __init__(self, config=None):
        """
        Initialize DistanceDetector.

        Args:
            config (dict, optional): Configuration options including:
                - distance_threshold (float): Threshold distance for alerts (default 1.0 meter)
        """
        super().__init__(config)
        self.config = config or {
            "distance_threshold": 1.0
        }

    def process_frame(self, detections=None):
        """
        Calculate distances between detected objects.

        Args:
            detections (list): List of positions [(x, y), ...] or bounding boxes.

        Returns:
            dict: Distances and alerts.
        """
        if not self.is_running:
            return {"distances": [], "alerts": []}

        distances = []
        alerts = []

        if not detections or len(detections) < 2:
            return {"distances": distances, "alerts": alerts}

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                p1 = np.array(detections[i])
                p2 = np.array(detections[j])
                dist = np.linalg.norm(p1 - p2)
                distances.append((i, j, dist))

                if dist < self.config["distance_threshold"]:
                    alerts.append((i, j, dist))
                    logger.debug(f"Distance alert between {i} and {j}: {dist:.2f}")

        return {"distances": distances, "alerts": alerts}

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Distance Detector",
            "description": "Estimates distances between detected objects or visitors",
            "config_options": {
                "distance_threshold": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Distance threshold for alerts (meters)"
                }
            }
        }
