 
"""
Motion detector for the VISIT Museum Tracker system.

This module implements motion detection using OpenCV's background subtraction
and contour analysis to identify movement in video frames.
"""

import cv2
import numpy as np
import time
from src.core.detector_base import DetectorBase


class MotionDetector(DetectorBase):
    """Motion detector using OpenCV background subtraction."""

    def __init__(self, config=None):
        """Initialize the motion detector with optional configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with settings:
                - history: Length of history for background subtractor (default: 100)
                - threshold: Threshold for motion detection (default: 30)
                - detect_shadows: Whether to detect shadows (default: False)
                - min_area: Minimum contour area to be considered motion (default: 500)
                - blur_size: Size of Gaussian blur kernel (default: (21, 21))
                - dilate_iterations: Number of dilation iterations (default: 2)
        """
        super().__init__(config)
        
        # Set default configuration if none is provided
        if not self.config:
            self.config = {
                "history": 100,
                "threshold": 30,
                "detect_shadows": False,
                "min_area": 500,
                "blur_size": (21, 21),
                "dilate_iterations": 2
            }
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get("history", 100),
            varThreshold=self.config.get("threshold", 30),
            detectShadows=self.config.get("detect_shadows", False)
        )
        
        # Initialize variables for tracking
        self.last_motion_time = time.time()
        self.motion_detected = False
        self.motion_regions = []
        
        # Initialize results storage
        self.results = {}
        
    def process_frame(self, frame):
        """Process a single frame to detect motion.
        
        Args:
            frame (numpy.ndarray): RGB image frame to process
            
        Returns:
            dict: Detection results containing:
                - motion_detected: Boolean indicating if motion was detected
                - motion_regions: List of contour coordinates
                - annotated_frame: Frame with motion visualization
        """
        if not self.is_running:
            return {"motion_detected": False, "motion_regions": [], "annotated_frame": frame.copy()}
        
        # Create a copy of the frame for visualization
        annotated_frame = frame.copy()
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur_size = self.config.get("blur_size", (21, 21))
        gray = cv2.GaussianBlur(gray, blur_size, 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Apply threshold to eliminate shadow detection if needed
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Dilate the threshold image to fill in holes
        dilate_iterations = self.config.get("dilate_iterations", 2)
        thresh = cv2.dilate(thresh, None, iterations=dilate_iterations)
        
        # Find contours in the threshold image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize motion regions
        self.motion_regions = []
        
        # Check for motion based on contour size
        min_area = self.config.get("min_area", 500)
        
        # Reset motion detection status
        self.motion_detected = False
        
        # Process contours
        for contour in contours:
            # If the contour is too small, ignore it
            if cv2.contourArea(contour) < min_area:
                continue
            
            # Motion detected
            self.motion_detected = True
            
            # Get bounding box coordinates
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Store region coordinates
            region = {
                "bbox": (x, y, x+w, y+h),
                "area": cv2.contourArea(contour),
                "center": (x + w//2, y + h//2)
            }
            self.motion_regions.append(region)
            
            # Draw the contour and bounding box on the frame
            cv2.drawContours(annotated_frame, [contour], 0, (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Update last motion time if motion is detected
        if self.motion_detected:
            self.last_motion_time = time.time()
        
        # Add text overlay with motion status
        status_text = "Motion Detected" if self.motion_detected else "No Motion"
        cv2.putText(annotated_frame, status_text, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add motion region count
        regions_text = f"Regions: {len(self.motion_regions)}"
        cv2.putText(annotated_frame, regions_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Store the results
        self.results = {
            "motion_detected": self.motion_detected,
            "motion_regions": self.motion_regions,
            "annotated_frame": annotated_frame
        }
        
        return self.results
    
    def reset_background(self):
        """Reset the background model.
        
        This is useful when the scene changes significantly.
        """
        # Re-initialize background subtractor with the same parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get("history", 100),
            varThreshold=self.config.get("threshold", 30),
            detectShadows=self.config.get("detect_shadows", False)
        )
    
    def set_sensitivity(self, value):
        """Set the motion detection sensitivity.
        
        Args:
            value (float): Sensitivity value from 0.0 to 1.0
        """
        # Map sensitivity to threshold (inverse relationship)
        # Higher sensitivity = lower threshold
        threshold = int(100 * (1.0 - value))
        threshold = max(10, min(threshold, 100))  # Clamp between 10 and 100
        
        # Update configuration
        self.config["threshold"] = threshold
        
        # Re-initialize background subtractor with new parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get("history", 100),
            varThreshold=threshold,
            detectShadows=self.config.get("detect_shadows", False)
        )
    
    def get_detector_info(self):
        """Get information about this detector.
        
        Returns:
            dict: Detector information
        """
        return {
            "name": "Motion Detector",
            "description": "Detects motion in video frames using background subtraction",
            "config_options": {
                "history": {
                    "type": "int",
                    "min": 2,
                    "max": 500,
                    "default": 100,
                    "description": "Length of history for background model"
                },
                "threshold": {
                    "type": "int",
                    "min": 10,
                    "max": 100,
                    "default": 30,
                    "description": "Threshold for motion detection"
                },
                "detect_shadows": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to detect shadows"
                },
                "min_area": {
                    "type": "int",
                    "min": 50,
                    "max": 10000,
                    "default": 500,
                    "description": "Minimum contour area to be considered motion"
                },
                "blur_size": {
                    "type": "tuple",
                    "options": [(5, 5), (11, 11), (21, 21), (31, 31)],
                    "default": (21, 21),
                    "description": "Size of Gaussian blur kernel"
                },
                "dilate_iterations": {
                    "type": "int",
                    "min": 0,
                    "max": 10,
                    "default": 2,
                    "description": "Number of dilation iterations"
                }
            }
        }