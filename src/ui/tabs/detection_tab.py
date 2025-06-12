"""
Detection tab for the VISIT Museum Tracker application.

This module implements the detection tab UI which displays the camera feed
and controls for various detectors.
"""

import os
import cv2
import logging
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QTabWidget, QGroupBox, QRadioButton, QSlider, QFileDialog,
    QMessageBox, QGridLayout, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor
from src.detectors.tracking.person_tracker import PersonTracker
from src.ui.components.video_player import VideoPlayer
from src.detectors.face.face_detector import FaceDetector
from src.detectors.face.expression_detector import ExpressionDetector
from src.detectors.zone_detector import ZoneDetector
from src.detectors.motion_detector import MotionDetector
from src.detectors.tracking.person_tracker import PersonTracker  # Add this for the person tracker
from datetime import datetime # Add the new person tracker
from src.utils.config import Config


try:
    from src.detectors.face.face_detector import FaceDetector
except ImportError:
    # Create a placeholder if the module doesn't exist
    class FaceDetector:
        def __init__(self, config=None):
            self.config = config or {}
            
        def process_frame(self, frame):
            return frame

try:
    from src.detectors.face.expression_detector import ExpressionDetector
except ImportError:
    # Create a placeholder if the module doesn't exist
    class ExpressionDetector:
        def __init__(self, config=None):
            self.config = config or {}
            
        def process_frame(self, frame):
            return frame

try:
    from src.detectors.zone_detector import ZoneDetector
except ImportError:
    # Create a placeholder if the module doesn't exist
    class ZoneDetector:
        def __init__(self, config=None):
            self.config = config or {}
            
        def process_frame(self, frame):
            return frame

try:
    from src.detectors.motion_detector import MotionDetector
except ImportError:
    # Create a placeholder if the module doesn't exist
    class MotionDetector:
        def __init__(self, config=None):
            self.config = config or {}
            
        def process_frame(self, frame):
            return frame

try:
    from src.detectors.tracking.person_tracker import PersonTracker
except ImportError:
    # Create a placeholder if the module doesn't exist
    class PersonTracker:
        def __init__(self, config=None):
            self.config = config or {}
            
        def process_frame(self, frame):
            return frame

from src.utils.config import Config


logger = logging.getLogger("DetectionTab")

class DetectionTab(QWidget):
    """Detection tab for the VISIT Museum Tracker application."""
    
    detection_updated = pyqtSignal(dict)
    
    def __init__(self, camera):
        """Initialize the detection tab.
        
        Args:
            camera: Camera instance for capturing video
        """
        super().__init__()
        
        # Store camera reference
        self.camera = camera
        
        # Load configuration
        self.config = Config()
        
        # Initialize detectors
        self.detectors = {}
        self.detector_enabled = {}
        self.detector_panels = {}
        self.init_detectors()
        
        # Initialize UI
        self.init_ui()
        
        # Start detectors
        self.start_detectors()


   
        
    def init_detectors(self):
        """Initialize detector instances."""
        # Create detector instances
        self.detectors["face"] = FaceDetector(self.config.get_detector_config("face"))
        self.detectors["expression"] = ExpressionDetector(self.config.get_detector_config("expression"))
        self.detectors["motion"] = MotionDetector(self.config.get_detector_config("motion"))
        self.detectors["zone"] = ZoneDetector(self.config.get_detector_config("zone"))
        self.detectors["person_tracker"] = PersonTracker(self.config.get_detector_config("person_tracker"))
        # Set camera reference        
        # Connect detectors
        if "zone" in self.detectors and "motion" in self.detectors:
            self.detectors["zone"].set_motion_detector(self.detectors["motion"])
        
        if "zone" in self.detectors and "face" in self.detectors:
            self.detectors["zone"].set_face_detector(self.detectors["face"])
            
        if "person_tracker" in self.detectors and "face" in self.detectors:
            self.detectors["person_tracker"].set_face_detector(self.detectors["face"])
        
        if "face" in self.detectors:
            self.detectors["person_tracker"].set_face_detector(self.detectors["face"])
        self.detector_enabled["person_tracker"] = self.config.is_detector_enabled("person_tracker")

        # Set default enabled state
        for name in self.detectors:
            self.detector_enabled[name] = self.config.is_detector_enabled(name)
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create video display area
        self.video_display = QLabel()
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_display)
        
        # Create detector controls
        detector_controls = QHBoxLayout()
        
        # Add detector checkboxes
        detector_controls.addWidget(QLabel("Detectors:"))
        
        for name in self.detectors:
            checkbox = QCheckBox(name.capitalize())
            checkbox.setChecked(self.detector_enabled[name])
            checkbox.stateChanged.connect(lambda state, n=name: self.toggle_detector(n, state))
            detector_controls.addWidget(checkbox)
        
        detector_controls.addStretch()
        main_layout.addLayout(detector_controls)
        
        # Create detector configuration tabs
        self.detector_tabs = QTabWidget()
        
        # Face detector panel
        if "face" in self.detectors:
            face_panel = self.create_face_detector_panel()
            self.detector_tabs.addTab(face_panel, "Face Detector")
            self.detector_panels["face"] = face_panel
        
        # Expression detector panel
        if "expression" in self.detectors:
            expression_panel = self.create_expression_detector_panel()
            self.detector_tabs.addTab(expression_panel, "Expression Detector")
            self.detector_panels["expression"] = expression_panel
        
        # Motion detector panel
        if "motion" in self.detectors:
            motion_panel = self.create_motion_detector_panel()
            self.detector_tabs.addTab(motion_panel, "Motion Detector")
            self.detector_panels["motion"] = motion_panel
        
        # Zone detector panel
        if "zone" in self.detectors:
            zone_panel = self.create_zone_detector_panel()
            self.detector_tabs.addTab(zone_panel, "Zone Detector")
            self.detector_panels["zone"] = zone_panel
            
        # Person tracker panel
        if "person_tracker" in self.detectors:
            tracker_panel = self.create_person_tracker_panel()
            self.detector_tabs.addTab(tracker_panel, "Person Tracker")
            self.detector_panels["person_tracker"] = tracker_panel
        
        main_layout.addWidget(self.detector_tabs)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def create_face_detector_panel(self):
        """Create the face detector configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Configuration options
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        # Detection confidence
        config_layout.addWidget(QLabel("Detection Confidence:"), 0, 0)
        confidence_slider = QSlider(Qt.Horizontal)
        confidence_slider.setMinimum(1)
        confidence_slider.setMaximum(10)
        confidence_slider.setValue(int(self.detectors["face"].config.get("min_detection_confidence", 0.5) * 10))
        confidence_slider.valueChanged.connect(
            lambda value: self.update_detector_config("face", "min_detection_confidence", value / 10.0))
        config_layout.addWidget(confidence_slider, 0, 1)
        confidence_label = QLabel(f"{confidence_slider.value() / 10.0:.1f}")
        confidence_slider.valueChanged.connect(lambda value: confidence_label.setText(f"{value / 10.0:.1f}"))
        config_layout.addWidget(confidence_label, 0, 2)
        
        # Model selection
        config_layout.addWidget(QLabel("Model:"), 1, 0)
        model_combo = QComboBox()
        model_combo.addItem("Short-range", 0)
        model_combo.addItem("Full-range", 1)
        model_combo.setCurrentIndex(self.detectors["face"].config.get("model_selection", 0))
        model_combo.currentIndexChanged.connect(
            lambda index: self.update_detector_config("face", "model_selection", model_combo.currentData()))
        config_layout.addWidget(model_combo, 1, 1, 1, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Status area
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.face_count_label = QLabel("Detected Faces: 0")
        status_layout.addWidget(self.face_count_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_expression_detector_panel(self):
        """Create the expression detector configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Configuration options
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        # Detection confidence
        config_layout.addWidget(QLabel("Detection Confidence:"), 0, 0)
        confidence_slider = QSlider(Qt.Horizontal)
        confidence_slider.setMinimum(1)
        confidence_slider.setMaximum(10)
        confidence_slider.setValue(int(self.detectors["expression"].config.get("min_detection_confidence", 0.5) * 10))
        confidence_slider.valueChanged.connect(
            lambda value: self.update_detector_config("expression", "min_detection_confidence", value / 10.0))
        config_layout.addWidget(confidence_slider, 0, 1)
        confidence_label = QLabel(f"{confidence_slider.value() / 10.0:.1f}")
        confidence_slider.valueChanged.connect(lambda value: confidence_label.setText(f"{value / 10.0:.1f}"))
        config_layout.addWidget(confidence_label, 0, 2)
        
        # Tracking confidence
        config_layout.addWidget(QLabel("Tracking Confidence:"), 1, 0)
        tracking_slider = QSlider(Qt.Horizontal)
        tracking_slider.setMinimum(1)
        tracking_slider.setMaximum(10)
        tracking_slider.setValue(int(self.detectors["expression"].config.get("min_tracking_confidence", 0.5) * 10))
        tracking_slider.valueChanged.connect(
            lambda value: self.update_detector_config("expression", "min_tracking_confidence", value / 10.0))
        config_layout.addWidget(tracking_slider, 1, 1)
        tracking_label = QLabel(f"{tracking_slider.value() / 10.0:.1f}")
        tracking_slider.valueChanged.connect(lambda value: tracking_label.setText(f"{value / 10.0:.1f}"))
        config_layout.addWidget(tracking_label, 1, 2)
        
        # Max faces
        config_layout.addWidget(QLabel("Max Faces:"), 2, 0)
        max_faces_slider = QSlider(Qt.Horizontal)
        max_faces_slider.setMinimum(1)
        max_faces_slider.setMaximum(5)
        max_faces_slider.setValue(self.detectors["expression"].config.get("max_num_faces", 1))
        max_faces_slider.valueChanged.connect(
            lambda value: self.update_detector_config("expression", "max_num_faces", value))
        config_layout.addWidget(max_faces_slider, 2, 1)
        max_faces_label = QLabel(f"{max_faces_slider.value()}")
        max_faces_slider.valueChanged.connect(lambda value: max_faces_label.setText(f"{value}"))
        config_layout.addWidget(max_faces_label, 2, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Enable expressions
        expressions_group = QGroupBox("Enable Expressions")
        expressions_layout = QVBoxLayout()
        
        expression_options = ["smile", "frown", "surprise", "neutral"]
        enabled_expressions = self.detectors["expression"].config.get("enable_expressions", expression_options.copy())
        
        for expression in expression_options:
            checkbox = QCheckBox(expression.capitalize())
            checkbox.setChecked(expression in enabled_expressions)
            checkbox.stateChanged.connect(lambda state, e=expression: self.toggle_expression(e, state))
            expressions_layout.addWidget(checkbox)
        
        expressions_group.setLayout(expressions_layout)
        layout.addWidget(expressions_group)
        
        # Status area
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.expression_label = QLabel("Current Expression: None")
        status_layout.addWidget(self.expression_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_motion_detector_panel(self):
        """Create the motion detector configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Configuration options
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        # Threshold
        config_layout.addWidget(QLabel("Threshold:"), 0, 0)
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setMinimum(1)
        threshold_slider.setMaximum(100)
        threshold_slider.setValue(int(self.detectors["motion"].config.get("threshold", 25)))
        threshold_slider.valueChanged.connect(
            lambda value: self.update_detector_config("motion", "threshold", value))
        config_layout.addWidget(threshold_slider, 0, 1)
        threshold_label = QLabel(f"{threshold_slider.value()}")
        threshold_slider.valueChanged.connect(lambda value: threshold_label.setText(f"{value}"))
        config_layout.addWidget(threshold_label, 0, 2)
        
        # Blur size
        config_layout.addWidget(QLabel("Blur Size:"), 1, 0)
        blur_slider = QSlider(Qt.Horizontal)
        blur_slider.setMinimum(1)
        blur_slider.setMaximum(21)
        blur_slider.setSingleStep(2)
        # Ensure blur size is odd
        blur_size = self.detectors["motion"].config.get("blur_size", 5)
        if blur_size % 2 == 0:
            blur_size += 1
        blur_slider.setValue(blur_size)
        blur_slider.valueChanged.connect(
            lambda value: self.update_detector_config("motion", "blur_size", value if value % 2 == 1 else value + 1))
        config_layout.addWidget(blur_slider, 1, 1)
        blur_label = QLabel(f"{blur_slider.value()}")
        blur_slider.valueChanged.connect(lambda value: blur_label.setText(f"{value if value % 2 == 1 else value + 1}"))
        config_layout.addWidget(blur_label, 1, 2)
        
        # Min area
        config_layout.addWidget(QLabel("Min Area:"), 2, 0)
        min_area_slider = QSlider(Qt.Horizontal)
        min_area_slider.setMinimum(100)
        min_area_slider.setMaximum(5000)
        min_area_slider.setSingleStep(100)
        min_area_slider.setValue(self.detectors["motion"].config.get("min_area", 500))
        min_area_slider.valueChanged.connect(
            lambda value: self.update_detector_config("motion", "min_area", value))
        config_layout.addWidget(min_area_slider, 2, 1)
        min_area_label = QLabel(f"{min_area_slider.value()}")
        min_area_slider.valueChanged.connect(lambda value: min_area_label.setText(f"{value}"))
        config_layout.addWidget(min_area_label, 2, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_motion_checkbox = QCheckBox("Show Motion Areas")
        self.show_motion_checkbox.setChecked(self.detectors["motion"].config.get("show_motion", True))
        self.show_motion_checkbox.stateChanged.connect(
            lambda state: self.update_detector_config("motion", "show_motion", state == Qt.Checked))
        display_layout.addWidget(self.show_motion_checkbox)
        
        self.show_threshold_checkbox = QCheckBox("Show Threshold Image")
        self.show_threshold_checkbox.setChecked(self.detectors["motion"].config.get("show_threshold", False))
        self.show_threshold_checkbox.stateChanged.connect(
            lambda state: self.update_detector_config("motion", "show_threshold", state == Qt.Checked))
        display_layout.addWidget(self.show_threshold_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Status area
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.motion_status_label = QLabel("Motion Detected: No")
        status_layout.addWidget(self.motion_status_label)
        
        self.motion_regions_label = QLabel("Motion Regions: 0")
        status_layout.addWidget(self.motion_regions_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_zone_detector_panel(self):
        """Create the zone detector configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Zone management
        zone_group = QGroupBox("Zone Management")
        zone_layout = QVBoxLayout()
        
        # Zone controls
        zone_controls = QHBoxLayout()
        
        self.add_zone_btn = QPushButton("Add Zone")
        self.add_zone_btn.clicked.connect(self.start_adding_zone)
        zone_controls.addWidget(self.add_zone_btn)
        
        self.cancel_zone_btn = QPushButton("Cancel")
        self.cancel_zone_btn.clicked.connect(self.cancel_adding_zone)
        self.cancel_zone_btn.setEnabled(False)
        zone_controls.addWidget(self.cancel_zone_btn)
        
        self.remove_zone_btn = QPushButton("Remove Zone")
        self.remove_zone_btn.clicked.connect(self.remove_selected_zone)
        zone_controls.addWidget(self.remove_zone_btn)
        
        zone_layout.addLayout(zone_controls)
        
        # Zone list
        self.zone_combo = QComboBox()
        self.zone_combo.currentIndexChanged.connect(self.select_zone)
        zone_layout.addWidget(self.zone_combo)
        
        # Zone info
        self.zone_info_label = QLabel("No zone selected")
        zone_layout.addWidget(self.zone_info_label)
        
        zone_group.setLayout(zone_layout)
        layout.addWidget(zone_group)
        
        # Zone drawing instructions
        drawing_group = QGroupBox("Drawing Instructions")
        drawing_layout = QVBoxLayout()
        
        instructions_label = QLabel(
            "1. Click 'Add Zone' to start drawing a new zone\n"
            "2. Click on the video to add points\n"
            "3. Press Enter or double-click to complete the zone\n"
            "4. Press Escape or click 'Cancel' to cancel"
        )
        drawing_layout.addWidget(instructions_label)
        
        drawing_group.setLayout(drawing_layout)
        layout.addWidget(drawing_group)
        
        # Zone status
        status_group = QGroupBox("Zone Status")
        status_layout = QVBoxLayout()
        
        self.zone_status_label = QLabel("No zones defined")
        status_layout.addWidget(self.zone_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Flag for zone drawing mode
        self.drawing_zone = False
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_person_tracker_panel(self):
        """Create the person tracker configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Configuration options
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout()
        
        # Max age
        config_layout.addWidget(QLabel("Max Track Age:"), 0, 0)
        max_age_slider = QSlider(Qt.Horizontal)
        max_age_slider.setMinimum(1)
        max_age_slider.setMaximum(100)
        max_age_slider.setValue(self.detectors["person_tracker"].config.get("max_age", 30))
        max_age_slider.valueChanged.connect(
            lambda value: self.update_detector_config("person_tracker", "max_age", value))
        config_layout.addWidget(max_age_slider, 0, 1)
        max_age_label = QLabel(f"{max_age_slider.value()}")
        max_age_slider.valueChanged.connect(lambda value: max_age_label.setText(f"{value}"))
        config_layout.addWidget(max_age_label, 0, 2)
        
        # Min hits
        config_layout.addWidget(QLabel("Min Hits to Confirm:"), 1, 0)
        min_hits_slider = QSlider(Qt.Horizontal)
        min_hits_slider.setMinimum(1)
        min_hits_slider.setMaximum(10)
        min_hits_slider.setValue(self.detectors["person_tracker"].config.get("min_hits", 3))
        min_hits_slider.valueChanged.connect(
            lambda value: self.update_detector_config("person_tracker", "min_hits", value))
        config_layout.addWidget(min_hits_slider, 1, 1)
        min_hits_label = QLabel(f"{min_hits_slider.value()}")
        min_hits_slider.valueChanged.connect(lambda value: min_hits_label.setText(f"{value}"))
        config_layout.addWidget(min_hits_label, 1, 2)
        
        # IoU threshold
        config_layout.addWidget(QLabel("IoU Threshold:"), 2, 0)
        iou_slider = QSlider(Qt.Horizontal)
        iou_slider.setMinimum(1)
        iou_slider.setMaximum(9)
        iou_slider.setValue(int(self.detectors["person_tracker"].config.get("iou_threshold", 0.3) * 10))
        iou_slider.valueChanged.connect(
            lambda value: self.update_detector_config("person_tracker", "iou_threshold", value / 10.0))
        config_layout.addWidget(iou_slider, 2, 1)
        iou_label = QLabel(f"{iou_slider.value() / 10.0:.1f}")
        iou_slider.valueChanged.connect(lambda value: iou_label.setText(f"{value / 10.0:.1f}"))
        config_layout.addWidget(iou_label, 2, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.show_trajectories_checkbox = QCheckBox("Show Trajectories")
        self.show_trajectories_checkbox.setChecked(self.detectors["person_tracker"].config.get("show_trajectories", True))
        self.show_trajectories_checkbox.stateChanged.connect(
            lambda state: self.update_detector_config("person_tracker", "show_trajectories", state == Qt.Checked))
        display_layout.addWidget(self.show_trajectories_checkbox)
        
        self.show_ids_checkbox = QCheckBox("Show IDs")
        self.show_ids_checkbox.setChecked(self.detectors["person_tracker"].config.get("show_ids", True))
        self.show_ids_checkbox.stateChanged.connect(
            lambda state: self.update_detector_config("person_tracker", "show_ids", state == Qt.Checked))
        display_layout.addWidget(self.show_ids_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Status area
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.active_tracks_label = QLabel("Active Tracks: 0")
        status_layout.addWidget(self.active_tracks_label)
        
        self.confirmed_tracks_label = QLabel("Confirmed Tracks: 0")
        status_layout.addWidget(self.confirmed_tracks_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Export controls
        export_group = QGroupBox("Data Export")
        export_layout = QVBoxLayout()
        
        export_btn = QPushButton("Export Tracking Data")
        export_btn.clicked.connect(self.export_tracking_data)
        export_layout.addWidget(export_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def start_detectors(self):
        """Start enabled detectors."""
        for name, detector in self.detectors.items():
            if self.detector_enabled[name]:
                detector.start()
                logger.info(f"Started detector: {name}")
    
    def stop_detectors(self):
        """Stop all detectors."""
        for name, detector in self.detectors.items():
            detector.stop()
        logger.info("Stopped all detectors")
    
    def toggle_detector(self, name, state):
        """Toggle a detector on or off.
        
        Args:
            name (str): Detector name
            state (int): Qt.Checked or Qt.Unchecked
        """
        enabled = state == Qt.Checked
        self.detector_enabled[name] = enabled
        
        if name in self.detectors:
            if enabled:
                self.detectors[name].start()
                logger.info(f"Started detector: {name}")
            else:
                self.detectors[name].stop()
                logger.info(f"Stopped detector: {name}")
                
        # Update configuration
        self.config.set_detector_config(name, {
            **self.detectors[name].config,
            "enabled": enabled
        })
        self.config.save()
    
    def update_detector_config(self, detector_name, key, value):
        """Update a detector configuration value.
        
        Args:
            detector_name (str): Name of the detector
            key (str): Configuration key
            value: New value
        """
        if detector_name in self.detectors:
            # Update detector config
            self.detectors[detector_name].config[key] = value
            
            # Update global config
            self.config.set_detector_config(detector_name, self.detectors[detector_name].config)
            self.config.save()
            
            logger.debug(f"Updated {detector_name} config: {key}={value}")
    
    def toggle_expression(self, expression, state):
        """Toggle an expression in the expression detector.
        
        Args:
            expression (str): Expression name
            state (int): Qt.Checked or Qt.Unchecked
        """
        if "expression" in self.detectors:
            enabled = state == Qt.Checked
            
            # Get current enabled expressions
            enabled_expressions = self.detectors["expression"].config.get("enable_expressions", [])
            
            # Update list
            if enabled and expression not in enabled_expressions:
                enabled_expressions.append(expression)
            elif not enabled and expression in enabled_expressions:
                enabled_expressions.remove(expression)
            
            # Update config
            self.update_detector_config("expression", "enable_expressions", enabled_expressions)
    
    def start_adding_zone(self):
        """Start adding a new zone."""
        if "zone" in self.detectors:
            self.drawing_zone = True
            self.detectors["zone"].start_drawing_zone()
            self.add_zone_btn.setEnabled(False)
            self.cancel_zone_btn.setEnabled(True)
            logger.info("Started adding zone")
    
    def cancel_adding_zone(self):
        """Cancel adding a new zone."""
        if "zone" in self.detectors:
            self.drawing_zone = False
            self.detectors["zone"].cancel_drawing_zone()
            self.add_zone_btn.setEnabled(True)
            self.cancel_zone_btn.setEnabled(False)
            logger.info("Cancelled adding zone")
    
    def finish_adding_zone(self):
        """Finish adding a new zone."""
        if "zone" in self.detectors:
            zone_id = self.detectors["zone"].finish_drawing_zone()
            if zone_id:
                self.drawing_zone = False
                self.add_zone_btn.setEnabled(True)
                self.cancel_zone_btn.setEnabled(False)
                logger.info(f"Added zone: {zone_id}")
                
                # Update zone list
                self.update_zone_list()
    
    def remove_selected_zone(self):
        """Remove the selected zone."""
        if "zone" in self.detectors and self.zone_combo.currentText():
            zone_id = self.zone_combo.currentText()
            if self.detectors["zone"].remove_zone(zone_id):
                logger.info(f"Removed zone: {zone_id}")
                
                # Update zone list
                self.update_zone_list()
    
    def select_zone(self):
        """Handle zone selection change."""
        if "zone" in self.detectors and self.zone_combo.currentText():
            zone_id = self.zone_combo.currentText()
            zone = self.detectors["zone"].get_zone(zone_id)
            if zone:
                self.zone_info_label.setText(f"Zone: {zone_id}")
                self.remove_zone_btn.setEnabled(True)
            else:
                self.zone_info_label.setText("No zone selected")
                self.remove_zone_btn.setEnabled(False)
    
    def update_zone_list(self):
        """Update the zone list."""
        if "zone" in self.detectors:
            self.zone_combo.clear()
            zones = self.detectors["zone"].get_zones()
            if zones:
                for zone_id in zones:
                    self.zone_combo.addItem(zone_id)
                self.zone_status_label.setText(f"Zones: {len(zones)}")
            else:
                self.zone_status_label.setText("No zones defined")
                self.zone_info_label.setText("No zone selected")
                self.remove_zone_btn.setEnabled(False)
    
    def export_tracking_data(self):
        """Export tracking data to a CSV file."""
        # Ask for export directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", "", QFileDialog.ShowDirsOnly
        )
        
        if not export_dir:
            return
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export trajectory data from database
        from src.utils.analytics_db import AnalyticsDB
        db = AnalyticsDB()
        
        # Get visitor paths for the last day
        trajectory_data = db.get_visitor_trajectories('day')
        
        if trajectory_data:
            # Create CSV file
            import csv
            csv_path = os.path.join(export_dir, f"visitor_tracks_{timestamp}.csv")
            
            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write header
                csv_writer.writerow(['timestamp', 'visitor_id', 'x', 'y'])
                
                # Write data
                for record_time, visitor_id, trajectory_json in trajectory_data:
                    try:
                        import json
                        trajectory = json.loads(trajectory_json)
                        for x, y in trajectory:
                            csv_writer.writerow([record_time, visitor_id, x, y])
                    except Exception as e:
                        logger.error(f"Error exporting trajectory: {e}")
            
            logger.info(f"Exported tracking data to {csv_path}")
            QMessageBox.information(self, "Export Complete", 
                                  f"Tracking data exported to:\n{csv_path}")
        else:
            QMessageBox.warning(self, "Export Failed", 
                              "No tracking data available to export")
    
    def process_frame(self, frame):
        """Process a frame with all enabled detectors.
        
        Args:
            frame (numpy.ndarray): Frame to process
            
        Returns:
            numpy.ndarray: Processed frame
        """
        if frame is None:
            return None
        
        # Make a copy of the frame to avoid modifying the original
        input_frame = frame.copy()
        
        # Process with each detector
        try:
            # Process face detector
            if "face" in self.detectors and self.detector_enabled["face"]:
                face_results = self.detectors["face"].process_frame(input_frame)
                if face_results and "annotated_frame" in face_results:
                    input_frame = face_results["annotated_frame"]
                    
                    # Update face count label
                    if "face_detections" in face_results:
                        self.face_count_label.setText(f"Detected Faces: {len(face_results['face_detections'])}")
            
            # Process expression detector
            if "expression" in self.detectors and self.detector_enabled["expression"]:
                expression_results = self.detectors["expression"].process_frame(input_frame)
                if expression_results and "annotated_frame" in expression_results:
                    input_frame = expression_results["annotated_frame"]
                    
                    # Update expression label
                    if "expressions" in expression_results and expression_results["expressions"]:
                        # Find most likely expression
                        max_expression = max(expression_results["expressions"].items(), key=lambda x: x[1])
                        expression_name, confidence = max_expression
                        self.expression_label.setText(f"Current Expression: {expression_name} ({confidence:.2f})")
            
            # Process motion detector
            motion_regions = None
            if "motion" in self.detectors and self.detector_enabled["motion"]:
                motion_results = self.detectors["motion"].process_frame(input_frame)
                if motion_results and "annotated_frame" in motion_results:
                    input_frame = motion_results["annotated_frame"]
                    
                    # Update motion status
                    if "motion_detected" in motion_results:
                        self.motion_status_label.setText(f"Motion Detected: {'Yes' if motion_results['motion_detected'] else 'No'}")
                    
                    # Update motion regions
                    if "motion_regions" in motion_results:
                        motion_regions = motion_results["motion_regions"]
                        self.motion_regions_label.setText(f"Motion Regions: {len(motion_regions)}")
            
            # Process zone detector
            if "zone" in self.detectors and self.detector_enabled["zone"]:
                if motion_regions is not None:
                    # Pass motion regions to zone detector if available
                    zone_results = self.detectors["zone"].process_frame(input_frame, motion_regions)
                else:
                    zone_results = self.detectors["zone"].process_frame(input_frame)
                    
                if zone_results and "annotated_frame" in zone_results:
                    input_frame = zone_results["annotated_frame"]
                    
                    # Update zone status
                    if "occupied_zones" in zone_results:
                        occupied_count = sum(1 for occupied in zone_results["occupied_zones"].values() if occupied)
                        total_count = len(zone_results["occupied_zones"])
                        self.zone_status_label.setText(f"Occupied Zones: {occupied_count}/{total_count}")
            
            # Process person tracker
            if "person_tracker" in self.detectors and self.detector_enabled["person_tracker"]:
                tracker_results = self.detectors["person_tracker"].process_frame(input_frame)
                if tracker_results and "annotated_frame" in tracker_results:
                    input_frame = tracker_results["annotated_frame"]
                    
                    # Update tracker status
                    if "tracks" in tracker_results:
                        tracks = tracker_results["tracks"]
                        active_tracks = len(tracks)
                        confirmed_tracks = sum(1 for track in tracks.values() if track.get("confirmed", False))
                        
                        self.active_tracks_label.setText(f"Active Tracks: {active_tracks}")
                        self.confirmed_tracks_label.setText(f"Confirmed Tracks: {confirmed_tracks}")
            
            


        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return input_frame
    
    def update_frame(self, frame):
        """Update the video display with a new frame.
        
        Args:
            frame (numpy.ndarray): Frame to display
        """
        if frame is None:
            return
        
        # Process the frame with detectors
        processed_frame = self.process_frame(frame)
        
        if processed_frame is not None:
            # Convert to QImage
            h, w, c = processed_frame.shape
            if c == 3:
                qimg = QImage(processed_frame.data, w, h, w * c, QImage.Format_RGB888)
                qimg = qimg.rgbSwapped()  # BGR to RGB
            else:
                qimg = QImage(processed_frame.data, w, h, w, QImage.Format_Grayscale8)
            
            # Display the image
            pixmap = QPixmap.fromImage(qimg)
            self.video_display.setPixmap(pixmap.scaled(
                self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def on_tab_selected(self):
        """Handle when the tab is selected."""
        # Make sure the camera is connected to the frame update
        if self.camera.is_running:
            self.camera.frame_ready.connect(self.update_frame)
            logger.debug("Connected camera to frame update")
    
    def update_tab(self):
        """Update the tab state."""
        if self.camera.is_running:
            # Get current frame
            frame = self.camera.get_frame()
            
            # Update the display
            self.update_frame(frame)
    
    def mousePressEvent(self, event):
        """Handle mouse press events.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        # Check if we're in zone drawing mode
        if self.drawing_zone and "zone" in self.detectors:
            # Get the position relative to the video display
            pos = self.video_display.mapFromParent(event.pos())
            
            # Check if the click is inside the video display
            if self.video_display.rect().contains(pos):
                # Get the position relative to the actual video frame
                pixmap = self.video_display.pixmap()
                if pixmap:
                    # Scale the position to the original video dimensions
                    scale_x = pixmap.width() / self.video_display.width()
                    scale_y = pixmap.height() / self.video_display.height()
                    
                    # Calculate the offset to center the image in the display
                    offset_x = (self.video_display.width() - pixmap.width() / scale_x) / 2
                    offset_y = (self.video_display.height() - pixmap.height() / scale_y) / 2
                    
                    # Calculate the scaled position
                    scaled_x = (pos.x() - offset_x) * scale_x
                    scaled_y = (pos.y() - offset_y) * scale_y
                    
                    # Add the point to the zone
                    self.detectors["zone"].add_point_to_zone((int(scaled_x), int(scaled_y)))
                    logger.debug(f"Added zone point: ({int(scaled_x)}, {int(scaled_y)})")
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click events.
        
        Args:
            event (QMouseEvent): Mouse event
        """
        # Check if we're in zone drawing mode
        if self.drawing_zone and "zone" in self.detectors:
            # Finish the zone
            self.finish_adding_zone()
    
    def keyPressEvent(self, event):
        """Handle key press events.
        
        Args:
            event (QKeyEvent): Key event
        """
        # Check if we're in zone drawing mode
        if self.drawing_zone and "zone" in self.detectors:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                # Finish the zone
                self.finish_adding_zone()
            elif event.key() == Qt.Key_Escape:
                # Cancel the zone
                self.cancel_adding_zone()