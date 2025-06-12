"""
Person tracker module for the VISIT Museum Tracker system.

Tracks multiple people across frames, associates detections over time.
"""

import cv2
import numpy as np
import logging

from src.core.detector_base import DetectorBase

logger = logging.getLogger("PersonTracker")

class PersonTracker(DetectorBase):
    """Tracks multiple persons using bounding boxes and tracking logic."""

    def __init__(self, config=None):
        """
        Initialize PersonTracker.

        Args:
            config (dict, optional): Configuration options including:
                - max_age (int): Maximum age of tracks (frames) before deletion
                - min_hits (int): Minimum hits to confirm a track
                - iou_threshold (float): IoU threshold for association
        """
        super().__init__(config)
        self.config = config or {
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "show_trajectories": True,
            "show_ids": True
        }

        self.next_id = 0
        self.tracks = {}  # track_id -> track info dict
        self.lost_tracks = {}

    def process_frame(self, frame, detections=None):
        """
        Update person tracks based on detections.

        Args:
            frame (np.ndarray): Current video frame.
            detections (list): List of bounding boxes [(xmin, ymin, xmax, ymax), ...]

        Returns:
            dict: Tracking results including active tracks and annotated frame.
        """
        if not self.is_running:
            return {"tracks": {}, "annotated_frame": frame.copy()}

        annotated_frame = frame.copy()

        if not detections:
            self._update_lost_tracks()
            self._draw_tracks(annotated_frame)
            return {"tracks": self.tracks, "annotated_frame": annotated_frame}

        matches, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detections)

        # Update matched tracks
        for track_id, det_idx in matches.items():
            bbox = detections[det_idx]
            track = self.tracks[track_id]
            track['bbox'] = bbox
            track['hits'] += 1
            track['age'] = 0
            track['confirmed'] = track['hits'] >= self.config['min_hits']

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = detections[det_idx]
            self.tracks[self.next_id] = {
                'bbox': bbox,
                'hits': 1,
                'age': 0,
                'confirmed': False,
                'trajectory': []
            }
            self.next_id += 1

        # Increment age for unmatched tracks
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track['age'] += 1

        # Remove old tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['age'] <= self.config['max_age']}

        # Update trajectories
        for track in self.tracks.values():
            if track['confirmed']:
                bbox = track['bbox']
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                track['trajectory'].append(center)
                if len(track['trajectory']) > 30:
                    track['trajectory'].pop(0)

        self._draw_tracks(annotated_frame)

        return {"tracks": self.tracks, "annotated_frame": annotated_frame}

    def _match_detections_to_tracks(self, detections):
        """
        Match detections to existing tracks using IoU.

        Returns:
            tuple: matches dict, unmatched detections list, unmatched tracks list.
        """
        iou_threshold = self.config['iou_threshold']
        matches = {}
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        if not self.tracks or not detections:
            return matches, unmatched_dets, unmatched_tracks

        iou_matrix = []
        for track_id in unmatched_tracks:
            track_bbox = self.tracks[track_id]['bbox']
            ious = [self._iou(track_bbox, det) for det in detections]
            iou_matrix.append(ious)

        iou_matrix = np.array(iou_matrix)

        while iou_matrix.size > 0:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < iou_threshold:
                break
            track_idx, det_idx = max_idx
            track_id = unmatched_tracks[track_idx]

            matches[track_id] = det_idx
            unmatched_dets.remove(det_idx)
            unmatched_tracks.remove(track_id)

            iou_matrix = np.delete(iou_matrix, track_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, det_idx, axis=1)

        return matches, unmatched_dets, unmatched_tracks

    def _iou(self, boxA, boxB):
        """Calculate Intersection over Union for two boxes."""
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
        """Draw bounding boxes, IDs, and trajectories on image."""
        for track_id, track in self.tracks.items():
            if not track.get('confirmed', False):
                continue
            bbox = track['bbox']
            xmin, ymin, xmax, ymax = bbox
            color = (0, 255, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            if self.config.get("show_ids", True):
                cv2.putText(image, f"ID {track_id}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.config.get("show_trajectories", True):
                pts = track.get('trajectory', [])
                for i in range(1, len(pts)):
                    cv2.line(image, pts[i-1], pts[i], color, 2)

    def _update_lost_tracks(self):
        """Increment age for all tracks and remove old ones."""
        for track in self.tracks.values():
            track['age'] += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['age'] <= self.config['max_age']}

    def get_detector_info(self):
        """Return metadata about this detector."""
        return {
            "name": "Person Tracker",
            "description": "Tracks multiple persons over time using bounding box IoU",
            "config_options": {
                "max_age": {
                    "type": "int",
                    "default": 30,
                    "description": "Max age in frames to keep lost tracks"
                },
                "min_hits": {
                    "type": "int",
                    "default": 3,
                    "description": "Minimum hits to confirm a track"
                },
                "iou_threshold": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.3,
                    "description": "IoU threshold for matching detections"
                },
                "show_trajectories": {
                    "type": "bool",
                    "default": True,
                    "description": "Show trajectory lines"
                },
                "show_ids": {
                    "type": "bool",
                    "default": True,
                    "description": "Show track IDs"
                }
            }
        }
