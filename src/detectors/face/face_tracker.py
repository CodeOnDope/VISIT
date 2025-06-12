 
"""
Face tracker module for the VISIT Museum Tracker system.

Tracks detected faces over time, assigns IDs, and maintains face state.
Extends DetectorBase to integrate with the detection pipeline.
"""

import cv2
import numpy as np
import logging
from collections import deque

from src.core.detector_base import DetectorBase

logger = logging.getLogger("FaceTracker")

class FaceTracker(DetectorBase):
    """Tracks faces over frames using simple IoU matching and smoothing."""

    def __init__(self, config=None):
        """
        Initialize FaceTracker.

        Args:
            config (dict, optional): Configuration options including:
                - max_track_lifetime (int): Number of frames to keep lost tracks (default 30)
                - iou_threshold (float): IoU threshold for matching detections to tracks (default 0.3)
        """
        super().__init__(config)
        self.config = config or {
            "max_track_lifetime": 30,
            "iou_threshold": 0.3
        }
        self.next_track_id = 0
        self.tracks = {}  # track_id -> track_info dict
        self.lost_tracks = {}

    def process_frame(self, frame, detections=None):
        """
        Track faces based on detections for the current frame.

        Args:
            frame (np.ndarray): The current video frame (BGR).
            detections (list): List of detected face bounding boxes [(xmin, ymin, xmax, ymax), ...]

        Returns:
            dict: Tracking results with:
                - 'tracks': dict of track_id -> bbox, confidence, and state
                - 'annotated_frame': frame with tracking overlays
        """
        if not self.is_running:
            return {"tracks": {}, "annotated_frame": frame.copy()}

        annotated_frame = frame.copy()

        # If no detections, update lost tracks
        if not detections:
            self._update_lost_tracks()
            self._draw_tracks(annotated_frame)
            return {"tracks": self.tracks, "annotated_frame": annotated_frame}

        matches, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detections)

        # Update matched tracks with detections
        for track_id, det_idx in matches.items():
            bbox = detections[det_idx]
            self.tracks[track_id]['bbox'] = bbox
            self.tracks[track_id]['lost'] = 0  # reset lost counter

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = detections[det_idx]
            self.tracks[self.next_track_id] = {
                'bbox': bbox,
                'lost': 0
            }
            self.next_track_id += 1

        # Increment lost counter for unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['lost'] += 1

        # Remove tracks lost for too long
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.config['max_track_lifetime']}

        # Draw all tracks
        self._draw_tracks(annotated_frame)

        return {"tracks": self.tracks, "annotated_frame": annotated_frame}

    def _match_detections_to_tracks(self, detections):
        """
        Match detected bounding boxes to existing tracks using IoU.

        Args:
            detections (list): List of bounding boxes [(xmin, ymin, xmax, ymax), ...]

        Returns:
            tuple:
                matches (dict): track_id -> detection index
                unmatched_detections (list): detection indices not matched
                unmatched_tracks (list): track ids not matched
        """
        iou_threshold = self.config['iou_threshold']
        matches = {}
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        if not self.tracks or not detections:
            return matches, unmatched_detections, unmatched_tracks

        # Compute IoU between detections and tracks
        iou_matrix = []
        for track_id in unmatched_tracks:
            track_bbox = self.tracks[track_id]['bbox']
            ious = [self._iou(track_bbox, det) for det in detections]
            iou_matrix.append(ious)

        iou_matrix = np.array(iou_matrix)

        # Greedy matching: highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < iou_threshold:
                break
            track_idx, det_idx = max_idx
            track_id = unmatched_tracks[track_idx]

            matches[track_id] = det_idx

            # Remove matched from unmatched
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_id)

            # Remove row and column
            iou_matrix = np.delete(iou_matrix, track_idx, 0)
            iou_matrix = np.delete(iou_matrix, det_idx, 1)
            unmatched_detections = [i for i in unmatched_detections if i != det_idx]
            unmatched_tracks = [t for t in unmatched_tracks if t != track_id]

        return matches, unmatched_detections, unmatched_tracks

    def _iou(self, boxA, boxB):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            boxA (tuple): (xmin, ymin, xmax, ymax)
            boxB (tuple): (xmin, ymin, xmax, ymax)

        Returns:
            float: IoU score.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    def _draw_tracks(self, image):
        """
        Draw tracking bounding boxes on the image.

        Args:
            image (np.ndarray): Image to draw on.
        """
        for track_id, track in self.tracks.items():
            bbox = track['bbox']
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f"ID {track_id}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Face Tracker",
            "description": "Tracks faces over time using IoU matching",
            "config_options": {
                "max_track_lifetime": {
                    "type": "int",
                    "default": 30,
                    "description": "Number of frames to keep lost tracks"
                },
                "iou_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.3,
                    "description": "IoU threshold for matching detections"
                }
            }
        }
