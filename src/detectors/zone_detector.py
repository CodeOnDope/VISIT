# E:\Personal Projects\VISIT-Museum-Tracker\src\detectors\zone_detector.py

import cv2
import numpy as np
import logging

from src.core.detector_base import DetectorBase

class Zone:
    """Represents a detection zone in the video frame."""
    
    def __init__(self, points, name=None):
        """Initialize a zone with a set of points.
        
        Args:
            points (list): List of (x, y) points defining the zone boundary
            name (str, optional): Zone name. Defaults to None.
        """
        self.points = np.array(points, dtype=np.int32)
        self.name = name if name else f"Zone_{id(self)}"
        self.color = (0, 255, 0)  # Default color: green
        self.contains_visitor = False
        self.visitor_count = 0
    
    def is_point_inside(self, point):
        """Check if a point is inside the zone.
        
        Args:
            point (tuple): (x, y) coordinates to check
            
        Returns:
            bool: True if the point is inside the zone, False otherwise
        """
        return cv2.pointPolygonTest(self.points, point, False) >= 0
    
    def update_status(self, contains_visitor, visitor_count=0):
        """Update the zone status.
        
        Args:
            contains_visitor (bool): Whether the zone contains a visitor
            visitor_count (int, optional): Number of visitors in the zone. Defaults to 0.
        """
        self.contains_visitor = contains_visitor
        self.visitor_count = visitor_count
        
        # Change color based on status
        if self.contains_visitor:
            self.color = (0, 0, 255)  # Red when occupied
        else:
            self.color = (0, 255, 0)  # Green when empty

class ZoneDetector(DetectorBase):
    """Detector for tracking visitor presence in defined zones."""
    
    def __init__(self, config=None):
        """Initialize the zone detector with optional configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with settings.
        """
        super().__init__(config)
        
        # Initialize zones dictionary
        self.zones = {}
        
        # Load zones from config if provided
        if self.config and "zones" in self.config:
            for zone_config in self.config["zones"]:
                if "points" in zone_config and "name" in zone_config:
                    self.add_zone(zone_config["points"], zone_config["name"])
        
        # Drawing state
        self.drawing = False
        self.current_zone_points = []
        
        # References to other detectors
        self.face_detector = None
        self.motion_detector = None
    
    def set_face_detector(self, face_detector):
        """Set reference to the face detector.
        
        Args:
            face_detector: Face detector instance
        """
        self.face_detector = face_detector
    
    def set_motion_detector(self, motion_detector):
        """Set reference to the motion detector.
        
        Args:
            motion_detector: Motion detector instance
        """
        self.motion_detector = motion_detector
    
    def add_zone(self, points, name=None):
        """Add a new zone to be monitored.
        
        Args:
            points (list): List of (x, y) points defining the zone
            name (str, optional): Zone name. Defaults to None.
            
        Returns:
            str: Zone ID
        """
        if len(points) < 3:
            self.logger.warning("Cannot create zone with less than 3 points")
            return None
        
        zone = Zone(points, name)
        zone_id = zone.name
        self.zones[zone_id] = zone
        self.logger.info(f"Added zone: {zone_id}")
        return zone_id
    
    def remove_zone(self, zone_id):
        """Remove a zone by ID.
        
        Args:
            zone_id (str): Zone ID to remove
            
        Returns:
            bool: True if removed, False otherwise
        """
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.logger.info(f"Removed zone: {zone_id}")
            return True
        return False
    
    def start_drawing_zone(self):
        """Start drawing a new zone."""
        self.drawing = True
        self.current_zone_points = []
        self.logger.debug("Started drawing zone")
    
    def add_point_to_zone(self, point):
        """Add a point to the current zone being drawn.
        
        Args:
            point (tuple): (x, y) coordinates to add
        """
        if self.drawing:
            self.current_zone_points.append(point)
            self.logger.debug(f"Added point to zone: {point}")
    
    def finish_drawing_zone(self, name=None):
        """Finish drawing the current zone and add it.
        
        Args:
            name (str, optional): Name for the zone. Defaults to None.
            
        Returns:
            str: Zone ID if created, None otherwise
        """
        if self.drawing and len(self.current_zone_points) >= 3:
            zone_id = self.add_zone(self.current_zone_points, name)
            self.drawing = False
            self.current_zone_points = []
            self.logger.info(f"Finished drawing zone: {zone_id}")
            return zone_id
        else:
            self.logger.warning("Cannot finish zone, need at least 3 points")
            return None
    
    def cancel_drawing_zone(self):
        """Cancel drawing the current zone."""
        self.drawing = False
        self.current_zone_points = []
        self.logger.debug("Cancelled drawing zone")
    
    def process_frame(self, frame, motion_regions=None):
        """Process a single frame to detect zone occupancy.
        
        Args:
            frame (numpy.ndarray): Video frame to process
            motion_regions (list, optional): Motion regions from the motion detector.
                Defaults to None.
            
        Returns:
            dict: Detection results containing:
                - zones: Dictionary of zones
                - occupied_zones: Dictionary mapping zone IDs to occupancy status
                - annotated_frame: Frame with zone visualizations
        """
        if not self.is_running or frame is None:
            return {
                "zones": self.zones,
                "occupied_zones": {},
                "annotated_frame": frame.copy() if frame is not None else None
            }
        
        # Create a copy of the frame for visualization
        annotated_frame = frame.copy()
        
        # Draw zones
        for zone_id, zone in self.zones.items():
            # Draw zone polygon
            cv2.polylines(annotated_frame, [zone.points], True, zone.color, 2)
            
            # Fill zone with semi-transparent color
            overlay = annotated_frame.copy()
            alpha = 0.2  # Transparency factor
            cv2.fillPoly(overlay, [zone.points], zone.color)
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Add zone name and visitor count
            centroid = np.mean(zone.points, axis=0, dtype=np.int32)
            cv2.putText(annotated_frame, f"{zone.name} ({zone.visitor_count})", 
                       (int(centroid[0]), int(centroid[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw zone being created
        if self.drawing and len(self.current_zone_points) > 0:
            # Draw points
            for point in self.current_zone_points:
                cv2.circle(annotated_frame, point, 5, (0, 255, 255), -1)
            
            # Draw lines connecting points
            for i in range(len(self.current_zone_points) - 1):
                cv2.line(annotated_frame, self.current_zone_points[i], 
                         self.current_zone_points[i+1], (0, 255, 255), 2)
            
            # Connect last point to first if we have at least 3 points
            if len(self.current_zone_points) >= 3:
                cv2.line(annotated_frame, self.current_zone_points[-1], 
                         self.current_zone_points[0], (0, 255, 255), 2)
        
        # Check zone occupancy
        occupied_zones = {}
        
        # Use motion regions if provided
        if motion_regions:
            for zone_id, zone in self.zones.items():
                zone_occupied = False
                visitor_count = 0
                
                for x, y, w, h in motion_regions:
                    # Use center of motion region
                    center_point = (x + w//2, y + h//2)
                    
                    if zone.is_point_inside(center_point):
                        zone_occupied = True
                        visitor_count += 1
                
                zone.update_status(zone_occupied, visitor_count)
                occupied_zones[zone_id] = zone_occupied
        
        # Or use face detector if available
        elif self.face_detector and hasattr(self.face_detector, 'results') and self.face_detector.results:
            detections = self.face_detector.results.get('detections', [])
            
            for zone_id, zone in self.zones.items():
                zone_occupied = False
                visitor_count = 0
                
                for face in detections:
                    x, y, w, h = face
                    # Use bottom center of face as position
                    position = (x + w//2, y + h)
                    
                    if zone.is_point_inside(position):
                        zone_occupied = True
                        visitor_count += 1
                
                zone.update_status(zone_occupied, visitor_count)
                occupied_zones[zone_id] = zone_occupied
        
        return {
            "zones": self.zones,
            "occupied_zones": occupied_zones,
            "annotated_frame": annotated_frame
        }
    
    def get_zones(self):
        """Get all defined zones.
        
        Returns:
            dict: Dictionary of zones
        """
        return self.zones
    
    def get_zone(self, zone_id):
        """Get a specific zone by ID.
        
        Args:
            zone_id (str): Zone ID to retrieve
            
        Returns:
            Zone: Zone object if found, None otherwise
        """
        return self.zones.get(zone_id)
    
    def get_detector_info(self):
        """Get information about this detector.
        
        Returns:
            dict: Detector information
        """
        return {
            "name": "Zone Detector",
            "description": "Detects visitor presence in defined zones",
            "config_options": {
                "zones": {
                    "type": "list",
                    "default": [],
                    "description": "List of predefined zones"
                }
            }
        }