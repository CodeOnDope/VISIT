# src/detectors/tracking/person_tracker.py

import cv2
import numpy as np
import logging
import time

from src.core.detector_base import DetectorBase
from src.utils.analytics_db import AnalyticsDB

class PersonTracker(DetectorBase):
    """Detector for tracking individual visitors across video frames."""
    
    def __init__(self, config=None):
        """Initialize the person tracker with optional configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with settings:
                - max_age: Maximum frames to keep a track without matching. Defaults to 30.
                - min_hits: Minimum hits before track is confirmed. Defaults to 3.
                - iou_threshold: IoU threshold for track matching. Defaults to 0.3.
        """
        super().__init__(config)
        
        # Set default configuration if none is provided
        if not self.config:
            self.config = {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3
            }
        
        # Initialize analytics database
        self.analytics_db = AnalyticsDB()
        self.camera_id = self.config.get("camera_id", "default")
        
        # Initialize tracking state
        self.next_id = 1
        self.tracks = {}  # Dictionary to store active tracks: id -> {bbox, features, age, etc.}
        self.max_age = self.config.get("max_age", 30)  
        self.min_hits = self.config.get("min_hits", 3)  
        self.iou_threshold = self.config.get("iou_threshold", 0.3)
        
        # Reference to face detector for detections
        self.face_detector = None
        
    def set_face_detector(self, face_detector):
        """Set reference to the face detector for obtaining detections.
        
        Args:
            face_detector: Reference to the face detector
        """
        self.face_detector = face_detector
        
    def process_frame(self, frame, detections=None):
        """Process a single frame to track people.
        
        Args:
            frame (numpy.ndarray): Video frame to process
            detections (list, optional): List of detection dictionaries with bbox and 
                other attributes. If None, tries to get detections from face detector.
                
        Returns:
            dict: Detection results containing:
                - tracks: Dictionary of active tracks
                - annotated_frame: Frame with track visualizations
        """
        if not self.is_running:
            return {"tracks": {}, "annotated_frame": frame.copy()}
        
        # Create a copy of the frame for visualization
        annotated_frame = frame.copy()
        
        # Get detections from face detector if not provided
        if detections is None and self.face_detector is not None:
            # Try multiple ways to get face detections based on your face detector implementation
            if hasattr(self.face_detector, 'results') and self.face_detector.results is not None:
                # MediaPipe style detections
                if hasattr(self.face_detector.results, 'detections'):
                    face_detections = self.face_detector.results.detections
                    detections = []
                    if face_detections:
                        h, w, _ = frame.shape
                        for detection in face_detections:
                            bbox = detection.location_data.relative_bounding_box
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            
                            detections.append({
                                "bbox": (x, y, width, height),
                                "confidence": detection.score[0]
                            })
                # OpenCV style detections
                elif isinstance(self.face_detector.results, dict) and 'face_detections' in self.face_detector.results:
                    face_detections = self.face_detector.results['face_detections']
                    detections = []
                    for det in face_detections:
                        if isinstance(det, dict) and 'bbox' in det:
                            detections.append(det)
                        elif isinstance(det, tuple) and len(det) == 4:
                            x, y, w, h = det
                            detections.append({
                                "bbox": (x, y, w, h),
                                "confidence": 1.0
                            })
                # Another common format
                elif isinstance(self.face_detector.results, dict) and 'detections' in self.face_detector.results:
                    face_detections = self.face_detector.results['detections']
                    detections = []
                    for det in face_detections:
                        if isinstance(det, tuple) and len(det) == 4:
                            x, y, w, h = det
                            detections.append({
                                "bbox": (x, y, w, h),
                                "confidence": 1.0
                            })
        
        if not detections:
            # No detections available, just draw existing tracks
            annotated_frame = self._draw_tracks(annotated_frame)
            return {"tracks": self.tracks, "annotated_frame": annotated_frame}
        
        # Predict new locations of existing tracks
        self._predict()
        
        # Update tracks with new detections
        self._update(detections)
        
        # Draw tracks on the frame
        annotated_frame = self._draw_tracks(annotated_frame)
        
        # Record analytics data for each track
        self._record_analytics_data()
        
        return {"tracks": self.tracks, "annotated_frame": annotated_frame}
        
    def _predict(self):
        """Predict new locations of tracks based on motion model."""
        # Simple linear motion model
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            # Increment age
            track["age"] += 1
            
            # Remove old tracks
            if track["age"] > self.max_age:
                del self.tracks[track_id]
                continue
                
            # Predict new position using velocity if available
            if "velocity" in track:
                x, y, w, h = track["bbox"]
                vx, vy = track["velocity"]
                track["bbox"] = (x + vx, y + vy, w, h)
    
    def _update(self, detections):
        """Update tracks with new detections."""
        if not self.tracks:
            # No existing tracks, create new ones
            for det in detections:
                self._init_track(det)
            return
            
        # Calculate IoU between existing tracks and new detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_indices = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_indices):
            track_bbox = self.tracks[track_id]["bbox"]
            for j, det in enumerate(detections):
                det_bbox = det["bbox"]
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Apply greedy assignment
        matched_indices = []
        unmatched_tracks = list(range(len(track_indices)))
        unmatched_detections = list(range(len(detections)))
        
        # Sort by IoU in descending order
        flat_indices = np.argsort(-iou_matrix.flatten())
        for idx in flat_indices:
            i = idx // len(detections)
            j = idx % len(detections)
            
            # Skip if already matched or below threshold
            if i in unmatched_tracks and j in unmatched_detections and iou_matrix[i, j] >= self.iou_threshold:
                matched_indices.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_detections.remove(j)
        
        # Update matched tracks
        for row, col in matched_indices:
            track_id = track_indices[row]
            detection = detections[col]
            self._update_track(track_id, detection)
        
        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            self._init_track(detections[idx])
        
        # Update unmatched tracks
        for idx in unmatched_tracks:
            track_id = track_indices[idx]
            self.tracks[track_id]["hits"] = max(0, self.tracks[track_id]["hits"] - 1)
            
    def _init_track(self, detection):
        """Initialize a new track."""
        self.tracks[self.next_id] = {
            "bbox": detection["bbox"],
            "confidence": detection.get("confidence", 1.0),
            "features": detection.get("features", None),
            "age": 1,
            "hits": 1,
            "velocity": (0, 0),
            "confirmed": False,
            "trajectory": [
                (detection["bbox"][0] + detection["bbox"][2]//2, 
                 detection["bbox"][1] + detection["bbox"][3])
            ],
            "last_update": time.time()
        }
        self.next_id += 1
    
    def _update_track(self, track_id, detection):
        """Update an existing track with new detection."""
        track = self.tracks[track_id]
        old_bbox = track["bbox"]
        new_bbox = detection["bbox"]
        
        # Calculate velocity
        old_x, old_y = old_bbox[0], old_bbox[1]
        new_x, new_y = new_bbox[0], new_bbox[1]
        vx = new_x - old_x
        vy = new_y - old_y
        track["velocity"] = (vx, vy)
        
        # Update track data
        track["bbox"] = new_bbox
        track["confidence"] = detection.get("confidence", track["confidence"])
        if "features" in detection and detection["features"] is not None:
            track["features"] = detection["features"]
        
        # Update track status
        track["age"] = 1
        track["hits"] += 1
        if track["hits"] >= self.min_hits:
            track["confirmed"] = True
        
        # Update trajectory
        center_point = (new_bbox[0] + new_bbox[2]//2, new_bbox[1] + new_bbox[3])
        track["trajectory"].append(center_point)
        
        # Keep trajectory at a reasonable length
        if len(track["trajectory"]) > 50:
            track["trajectory"] = track["trajectory"][-50:]
            
        # Update last update time
        track["last_update"] = time.time()
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to x1, y1, x2, y2 format
        x1_1, y1_1 = x1, y1
        x2_1, y2_1 = x1 + w1, y1 + h1
        x1_2, y1_2 = x2, y2
        x2_2, y2_2 = x2 + w2, y2 + h2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / max(union_area, 1e-6)
    
    def _draw_tracks(self, frame):
        """Draw tracks on the frame."""
        for track_id, track in self.tracks.items():
            if not track.get("confirmed", False):
                continue
                
            x, y, w, h = [int(v) for v in track["bbox"]]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for confirmed tracks
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw ID
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory if available
            if "trajectory" in track and len(track["trajectory"]) > 1:
                points = np.array(track["trajectory"], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, color, 2)
                
        return frame
    
    def _record_analytics_data(self):
        """Record analytics data for tracks."""
        for track_id, track in self.tracks.items():
            if track.get("confirmed", False):
                # Record visitor path
                if "trajectory" in track and len(track["trajectory"]) > 0:
                    position = track["trajectory"][-1]
                    self.analytics_db.add_visitor_path(str(track_id), position, self.camera_id)
    
    def get_detector_info(self):
        """Get information about this detector."""
        return {
            "name": "Person Tracker",
            "description": "Tracks individual visitors across frames",
            "config_options": {
                "max_age": {
                    "type": "int",
                    "min": 1,
                    "max": 100,
                    "default": 30,
                    "description": "Maximum frames to keep a track without matching"
                },
                "min_hits": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                    "description": "Minimum hits before track is confirmed"
                },
                "iou_threshold": {
                    "type": "float",
                    "min": 0.1,
                    "max": 0.9,
                    "default": 0.3,
                    "description": "IoU threshold for track matching"
                },
                "camera_id": {
                    "type": "string",
                    "default": "default",
                    "description": "Camera identifier for analytics"
                }
            }
        }