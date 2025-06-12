# src/detectors/exhibit_detector.py
import cv2
import numpy as np
import logging
import time

from src.core.detector_base import DetectorBase
from src.utils.analytics_db import AnalyticsDB

logger = logging.getLogger("ExhibitDetector")

class Exhibit:
    """Represents an exhibit in the museum."""
    
    def __init__(self, points, name=None, description=None):
        self.points = np.array(points, dtype=np.int32)
        self.name = name if name else f"Exhibit_{id(self)}"
        self.description = description
        self.color = (0, 255, 0)  # Default color: green
        self.visitor_count = 0
        self.total_interactions = 0
        self.active_visitors = set()
    
    def is_point_inside(self, point):
        """Check if a point is inside the exhibit area."""
        return cv2.pointPolygonTest(self.points, point, False) >= 0
    
    def update_status(self, visitor_count, active_visitors=None):
        """Update the exhibit status."""
        self.visitor_count = visitor_count
        if active_visitors is not None:
            # Track new visitors
            new_visitors = active_visitors - self.active_visitors
            self.total_interactions += len(new_visitors)
            self.active_visitors = active_visitors

class ExhibitDetector(DetectorBase):
    """Detector for tracking visitor interactions with exhibits."""

    def __init__(self, config=None):
        super().__init__(config)
        
        if not self.config:
            self.config = {}
        
        # Initialize analytics database
        self.analytics_db = AnalyticsDB()
        self.camera_id = self.config.get("camera_id", "default")
        
        # Initialize exhibits dictionary
        self.exhibits = {}
        
        # Load exhibits if provided in config
        exhibits_config = self.config.get("exhibits", [])
        for exhibit_config in exhibits_config:
            if "points" in exhibit_config and "name" in exhibit_config:
                self.add_exhibit(
                    exhibit_config["points"], 
                    exhibit_config["name"],
                    exhibit_config.get("description", "")
                )
        
        # Initialize drawing state
        self.drawing = False
        self.current_exhibit_points = []
        
        # Add a database table for exhibit interactions
        self._init_db_table()
    
    def _init_db_table(self):
        """Initialize the database table for exhibit interactions."""
        try:
            conn = self.analytics_db.conn
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS exhibit_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    exhibit_id TEXT NOT NULL,
                    visitor_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0,
                    camera_id TEXT DEFAULT 'default'
                )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating exhibit interactions table: {e}")
    
    def add_exhibit(self, points, name=None, description=None):
        """Add a new exhibit to be monitored."""
        if len(points) < 3:
            logger.warning("Cannot create exhibit with less than 3 points")
            return None
        
        exhibit = Exhibit(points, name, description)
        exhibit_id = exhibit.name
        self.exhibits[exhibit_id] = exhibit
        logger.info(f"Added exhibit: {exhibit_id}")
        return exhibit_id
    
    def process_frame(self, frame, tracks=None):
        """Process a single frame to detect visitor interactions with exhibits."""
        if not self.is_running:
            return {"exhibits": self.exhibits, "annotated_frame": frame.copy()}
        
        # Create a copy of the frame for visualization
        annotated_frame = frame.copy()
        
        # Draw exhibits
        for exhibit_id, exhibit in self.exhibits.items():
            # Draw exhibit boundary
            cv2.polylines(annotated_frame, [exhibit.points], True, exhibit.color, 2)
            
            # Fill with semi-transparent color
            overlay = annotated_frame.copy()
            alpha = 0.2
            cv2.fillPoly(overlay, [exhibit.points], exhibit.color)
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Add exhibit name and visitor count
            centroid = np.mean(exhibit.points, axis=0, dtype=np.int32)
            cv2.putText(annotated_frame, f"{exhibit.name} ({exhibit.visitor_count})", 
                       (int(centroid[0]), int(centroid[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw exhibit being created
        if self.drawing and len(self.current_exhibit_points) > 0:
            # Draw points
            for point in self.current_exhibit_points:
                cv2.circle(annotated_frame, point, 5, (0, 255, 255), -1)
            
            # Draw lines connecting points
            for i in range(len(self.current_exhibit_points) - 1):
                cv2.line(annotated_frame, self.current_exhibit_points[i], 
                         self.current_exhibit_points[i+1], (0, 255, 255), 2)
            
            # Connect last point to first if we have at least 3 points
            if len(self.current_exhibit_points) >= 3:
                cv2.line(annotated_frame, self.current_exhibit_points[-1], 
                         self.current_exhibit_points[0], (0, 255, 255), 2)
        
        # Process visitor interactions if tracks are provided
        if tracks:
            self._process_interactions(tracks)
        
        return {"exhibits": self.exhibits, "annotated_frame": annotated_frame}
    
    def _process_interactions(self, tracks):
        """Process visitor interactions with exhibits."""
        # Reset visitor counts
        for exhibit in self.exhibits.values():
            exhibit.visitor_count = 0
        
        # Check each track against each exhibit
        for track_id, track in tracks.items():
            if not track.get("confirmed", False):
                continue
                
            # Get the bottom center of the person
            x, y, w, h = track["bbox"]
            person_point = (x + w//2, y + h)
            
            # Check each exhibit
            for exhibit_id, exhibit in self.exhibits.items():
                if exhibit.is_point_inside(person_point):
                    # Increment visitor count
                    exhibit.visitor_count += 1
                    
                    # Add to active visitors
                    active_visitors = exhibit.active_visitors.copy()
                    active_visitors.add(track_id)
                    
                    # Update exhibit status
                    exhibit.update_status(exhibit.visitor_count, active_visitors)
                    
                    # Record interaction in database
                    self._record_interaction(track_id, exhibit_id, "presence")
    
    def _record_interaction(self, visitor_id, exhibit_id, interaction_type):
        """Record an interaction in the database."""
        try:
            conn = self.analytics_db.conn
            if conn:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO exhibit_interactions (timestamp, exhibit_id, visitor_id, interaction_type, camera_id) VALUES (?, ?, ?, ?, ?)",
                    (timestamp, exhibit_id, str(visitor_id), interaction_type, self.camera_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording exhibit interaction: {e}")