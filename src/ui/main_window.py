"""
Main window for the VISIT Museum Tracker application.

This module implements the main application window and UI framework for the
VISIT Museum Tracker system.
"""

import os
import sys
import logging
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QAction, QToolBar, QFileDialog,
    QMessageBox, QComboBox, QCheckBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QPixmap

from src.ui.components.video_player import VideoPlayer
from src.ui.tabs.detection_tab import DetectionTab
from src.ui.tabs.settings_tab import SettingsTab
from src.ui.tabs.analytics_tab import AnalyticsTab
from src.ui.tabs.calibration_tab import CalibrationTab
from src.ui.tabs.media_tab import MediaTab
from src.ui.tabs.debug_tab import DebugTab
from src.core.camera import Camera
from src.utils.logger import setup_logger
from src.utils.config import Config


class MainWindow(QMainWindow):
    """Main application window for the VISIT Museum Tracker system."""

    def __init__(self, app):
        """Initialize the main window.
        
        Args:
            app (QApplication): The Qt application instance
        """
        super().__init__()
        
        # Store reference to the application
        self.app = app
        
        # Set up logging
        self.logger = setup_logger("MainWindow", level=logging.INFO)
        self.logger.info("Initializing main window")
        
        # Load configuration
        self.config = Config()
        self.config.load_defaults()
        
        # Set up camera
        camera_id = self.config.get("camera", "camera_id", 0)
        resolution = self.config.get("camera", "resolution", (640, 480))
        self.camera = Camera(camera_id=camera_id, resolution=resolution)
        
        # Initialize UI components
        self._init_ui()
        
        # Set up timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(50)  # 20 FPS update rate
        
        # Start the camera
        self._start_camera()
        
        self.logger.info("Main window initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("VISIT Museum Tracker")
        self.setMinimumSize(1024, 768)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create and add tabs
        self.detection_tab = DetectionTab(self.camera)
        self.settings_tab = SettingsTab(self.config)
        self.analytics_tab = AnalyticsTab()
        self.calibration_tab = CalibrationTab(self.camera)
        self.media_tab = MediaTab()
        self.debug_tab = DebugTab(self.logger)
        
        self.tab_widget.addTab(self.detection_tab, "Detection")
        self.tab_widget.addTab(self.analytics_tab, "Analytics")
        self.tab_widget.addTab(self.media_tab, "Media")
        self.tab_widget.addTab(self.calibration_tab, "Calibration")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.debug_tab, "Debug")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status bar elements
        self.status_camera = QLabel("Camera: Disconnected")
        self.status_fps = QLabel("FPS: 0")
        self.status_detections = QLabel("Detections: 0")
        
        self.status_bar.addPermanentWidget(self.status_camera)
        self.status_bar.addPermanentWidget(self.status_fps)
        self.status_bar.addPermanentWidget(self.status_detections)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Connect signals and slots
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        
        # Final setup
        self.status_bar.showMessage("Ready")
    
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # File actions
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video_file)
        file_menu.addAction(open_action)
        
        export_action = QAction("&Export Data...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.settings_tab))
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera menu
        camera_menu = menu_bar.addMenu("&Camera")
        
        # Camera actions
        start_camera_action = QAction("&Start Camera", self)
        start_camera_action.triggered.connect(self._start_camera)
        camera_menu.addAction(start_camera_action)
        
        stop_camera_action = QAction("S&top Camera", self)
        stop_camera_action.triggered.connect(self._stop_camera)
        camera_menu.addAction(stop_camera_action)
        
        camera_menu.addSeparator()
        
        calibrate_action = QAction("&Calibrate Camera", self)
        calibrate_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.calibration_tab))
        camera_menu.addAction(calibrate_action)
        
        # Detector menu
        detector_menu = menu_bar.addMenu("&Detectors")
        
        # Add detectors dynamically
        face_detector_action = QAction("&Face Detector", self)
        face_detector_action.setCheckable(True)
        face_detector_action.setChecked(True)
        detector_menu.addAction(face_detector_action)
        
        expression_detector_action = QAction("&Expression Detector", self)
        expression_detector_action.setCheckable(True)
        expression_detector_action.setChecked(True)
        detector_menu.addAction(expression_detector_action)
        
        motion_detector_action = QAction("&Motion Detector", self)
        motion_detector_action.setCheckable(True)
        motion_detector_action.setChecked(True)
        detector_menu.addAction(motion_detector_action)
        
        zone_detector_action = QAction("&Zone Detector", self)
        zone_detector_action.setCheckable(True)
        zone_detector_action.setChecked(True)
        detector_menu.addAction(zone_detector_action)
        
        # Add the new person tracker action
        person_tracker_action = QAction("&Person Tracker", self)
        person_tracker_action.setCheckable(True)
        person_tracker_action.setChecked(True)
        detector_menu.addAction(person_tracker_action)
        
        # Analytics menu
        analytics_menu = menu_bar.addMenu("&Analytics")
        
        view_analytics_action = QAction("&View Analytics", self)
        view_analytics_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.analytics_tab))
        analytics_menu.addAction(view_analytics_action)
        
        analytics_menu.addSeparator()
        
        export_analytics_action = QAction("&Export Analytics...", self)
        export_analytics_action.triggered.connect(self._export_analytics)
        analytics_menu.addAction(export_analytics_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)
        
        # Camera control buttons
        start_camera_action = QAction("Start Camera", self)
        start_camera_action.triggered.connect(self._start_camera)
        toolbar.addAction(start_camera_action)
        
        stop_camera_action = QAction("Stop Camera", self)
        stop_camera_action.triggered.connect(self._stop_camera)
        toolbar.addAction(stop_camera_action)
        
        toolbar.addSeparator()
        
        # Camera selection
        camera_label = QLabel("Camera:")
        toolbar.addWidget(camera_label)
        
        self.camera_selector = QComboBox()
        self.camera_selector.addItem("Camera 0", 0)
        self.camera_selector.addItem("Camera 1", 1)
        self.camera_selector.currentIndexChanged.connect(self._change_camera)
        toolbar.addWidget(self.camera_selector)
        
        toolbar.addSeparator()
        
        # Detector toggles
        detector_label = QLabel("Detectors:")
        toolbar.addWidget(detector_label)
        
        face_detector_check = QCheckBox("Face")
        face_detector_check.setChecked(True)
        toolbar.addWidget(face_detector_check)
        
        expression_detector_check = QCheckBox("Expression")
        expression_detector_check.setChecked(True)
        toolbar.addWidget(expression_detector_check)
        
        zone_detector_check = QCheckBox("Zone")
        zone_detector_check.setChecked(True)
        toolbar.addWidget(zone_detector_check)
        
        # Add the new person tracker checkbox
        person_tracker_check = QCheckBox("Track Visitors")
        person_tracker_check.setChecked(True)
        toolbar.addWidget(person_tracker_check)
        
        toolbar.addSeparator()
        
        # Snapshot button
        snapshot_action = QAction("Take Snapshot", self)
        snapshot_action.triggered.connect(self._take_snapshot)
        toolbar.addAction(snapshot_action)
    
    def _update_ui(self):
        """Update UI elements with current state."""
        # Update status bar
        if self.camera.is_running:
            camera_props = self.camera.get_properties()
            fps = camera_props.get("fps", 0)
            self.status_camera.setText(f"Camera: Connected ({camera_props.get('width', 0)}x{camera_props.get('height', 0)})")
            self.status_fps.setText(f"FPS: {fps:.1f}")
        else:
            self.status_camera.setText("Camera: Disconnected")
            self.status_fps.setText("FPS: 0")
        
        # Update current tab
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, "update_tab"):
            current_tab.update_tab()
    
    def _start_camera(self):
        """Start the camera."""
        if not self.camera.is_running:
            if self.camera.start():
                self.logger.info("Camera started")
                self.status_bar.showMessage("Camera started", 3000)
            else:
                self.logger.error("Failed to start camera")
                self.status_bar.showMessage("Failed to start camera", 3000)
                QMessageBox.warning(self, "Camera Error", 
                                   "Failed to start camera. Please check the camera connection.")
    
    def _stop_camera(self):
        """Stop the camera."""
        if self.camera.is_running:
            self.camera.stop()
            self.logger.info("Camera stopped")
            self.status_bar.showMessage("Camera stopped", 3000)
    
    def _change_camera(self, index):
        """Change the active camera.
        
        Args:
            index (int): Index of the selected camera in the combobox
        """
        camera_id = self.camera_selector.itemData(index)
        
        # Stop current camera
        self._stop_camera()
        
        # Update camera ID
        self.camera.camera_id = camera_id
        
        # Update configuration
        self.config.set("camera", "camera_id", camera_id)
        self.config.save()
        
        # Start new camera
        self._start_camera()
    
    def _open_video_file(self):
        """Open a video file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.logger.info(f"Opening video file: {file_path}")
            self.status_bar.showMessage(f"Opening video file: {os.path.basename(file_path)}", 3000)
            
            # Stop live camera
            self._stop_camera()
            
            # TODO: Implement video file loading in the camera class
            # For now, just show a message
            QMessageBox.information(self, "Open Video", 
                                   f"Video file support is coming soon!\n\nSelected: {file_path}")
    
    def _export_data(self):
        """Export detection data."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", ""
        )
        
        if directory:
            self.logger.info(f"Exporting data to: {directory}")
            self.status_bar.showMessage(f"Exporting data to: {os.path.basename(directory)}", 3000)
            
            # TODO: Implement data export
            # For now, just show a message
            QMessageBox.information(self, "Export Data", 
                                   f"Data export is coming soon!\n\nSelected directory: {directory}")
    
    def _export_analytics(self):
        """Export analytics data."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", ""
        )
        
        if directory:
            self.logger.info(f"Exporting analytics to: {directory}")
            self.status_bar.showMessage(f"Exporting analytics to: {os.path.basename(directory)}", 3000)
            
            # TODO: Implement analytics export
            # For now, just show a message
            QMessageBox.information(self, "Export Analytics", 
                                   f"Analytics export is coming soon!\n\nSelected directory: {directory}")
    
    def _take_snapshot(self):
        """Take a snapshot of the current camera frame."""
        if not self.camera.is_running:
            QMessageBox.warning(self, "Snapshot Error", "Camera is not running")
            return
        
        frame = self.camera.get_frame()
        if frame is None:
            QMessageBox.warning(self, "Snapshot Error", "Failed to capture frame")
            return
        
        # Save snapshot
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "", "Image Files (*.png *.jpg);;All Files (*)"
        )
        
        if file_path:
            # Save the image
            success = cv2.imwrite(file_path, frame)
            
            if success:
                self.logger.info(f"Snapshot saved to: {file_path}")
                self.status_bar.showMessage(f"Snapshot saved to: {os.path.basename(file_path)}", 3000)
            else:
                self.logger.error(f"Failed to save snapshot to: {file_path}")
                self.status_bar.showMessage("Failed to save snapshot", 3000)
                QMessageBox.warning(self, "Snapshot Error", "Failed to save snapshot")
    
    def _show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(self, "About VISIT Museum Tracker",
            "<h1>VISIT Museum Tracker</h1>"
            "<p>Version 0.1.0</p>"
            "<p>A computer vision system for museum visitor analytics.</p>"
            "<p>Copyright &copy; 2025 VISIT Museum Tracker</p>"
        )
    
    def _on_tab_changed(self, index):
        """Handle tab change event.
        
        Args:
            index (int): Index of the selected tab
        """
        tab_name = self.tab_widget.tabText(index)
        self.logger.info(f"Tab changed to: {tab_name}")
        
        # Update the current tab
        current_tab = self.tab_widget.widget(index)
        if hasattr(current_tab, "on_tab_selected"):
            current_tab.on_tab_selected()
    
    def closeEvent(self, event):
        """Handle window close event.
        
        Args:
            event (QCloseEvent): Close event
        """
        # Stop the camera
        self._stop_camera()
        
        # Stop update timer
        self.update_timer.stop()
        
        # Save configuration
        self.config.save()
        
        self.logger.info("Application closing")
        event.accept()