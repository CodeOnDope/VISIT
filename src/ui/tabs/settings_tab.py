"""
Settings tab for the VISIT Museum Tracker application.

This module implements the settings tab UI that allows users to configure
various aspects of the application.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QFileDialog, QTabWidget, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot

from src.utils.logger import setup_logger


class SettingsTab(QWidget):
    """Tab for application settings and configuration."""
    
    def __init__(self, config, parent=None):
        """Initialize the settings tab.
        
        Args:
            config (Config): Configuration manager instance
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Store configuration reference
        self.config = config
        
        # Set up logging
        self.logger = setup_logger("SettingsTab", level=logging.INFO)
        
        # Set up UI
        self._init_ui()
        
        # Load current settings
        self._load_settings()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget for different settings categories
        self.settings_tabs = QTabWidget()
        
        # Create and add settings tabs
        self.general_tab = self._create_general_tab()
        self.camera_tab = self._create_camera_tab()
        self.detector_tab = self._create_detector_tab()
        self.analytics_tab = self._create_analytics_tab()
        self.advanced_tab = self._create_advanced_tab()
        
        self.settings_tabs.addTab(self.general_tab, "General")
        self.settings_tabs.addTab(self.camera_tab, "Camera")
        self.settings_tabs.addTab(self.detector_tab, "Detectors")
        self.settings_tabs.addTab(self.analytics_tab, "Analytics")
        self.settings_tabs.addTab(self.advanced_tab, "Advanced")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.settings_tabs)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Add save and reset buttons
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self._save_settings)
        buttons_layout.addWidget(self.save_button)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_settings)
        buttons_layout.addWidget(self.reset_button)
        
        # Add spacer to push buttons to the right
        buttons_layout.addStretch()
        
        # Add buttons layout to main layout
        main_layout.addLayout(buttons_layout)
    
    def _create_general_tab(self):
        """Create general settings tab.
        
        Returns:
            QWidget: The general settings widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # User interface group
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout(ui_group)
        
        # Theme selector
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Default", "Dark", "Light"])
        ui_layout.addRow("Theme:", self.theme_combo)
        
        # Language selector
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Español", "Français", "Deutsch"])
        ui_layout.addRow("Language:", self.language_combo)
        
        layout.addWidget(ui_group)
        
        # Application behavior group
        behavior_group = QGroupBox("Application Behavior")
        behavior_layout = QFormLayout(behavior_group)
        
        # Start behavior
        self.start_camera_check = QCheckBox("Start camera on application launch")
        behavior_layout.addRow("", self.start_camera_check)
        
        # Auto-save
        self.auto_save_check = QCheckBox("Auto-save settings on exit")
        behavior_layout.addRow("", self.auto_save_check)
        
        layout.addWidget(behavior_group)
        
        # Add stretch to push content to the top
        layout.addStretch()
        
        return tab
    
    def _create_camera_tab(self):
        """Create camera settings tab.
        
        Returns:
            QWidget: The camera settings widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Camera selection group
        camera_group = QGroupBox("Camera Selection")
        camera_layout = QFormLayout(camera_group)
        
        # Camera source selector
        self.camera_source_combo = QComboBox()
        self.camera_source_combo.addItem("Camera 0", 0)
        self.camera_source_combo.addItem("Camera 1", 1)
        camera_layout.addRow("Camera Source:", self.camera_source_combo)
        
        # Resolution selector
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("640 x 480", (640, 480))
        self.resolution_combo.addItem("800 x 600", (800, 600))
        self.resolution_combo.addItem("1280 x 720", (1280, 720))
        self.resolution_combo.addItem("1920 x 1080", (1920, 1080))
        camera_layout.addRow("Resolution:", self.resolution_combo)
        
        # FPS selector
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        camera_layout.addRow("Target FPS:", self.fps_spin)
        
        layout.addWidget(camera_group)
        
        # Video processing group
        video_group = QGroupBox("Video Processing")
        video_layout = QFormLayout(video_group)
        
        # Flip horizontal
        self.flip_h_check = QCheckBox("Flip horizontally")
        video_layout.addRow("", self.flip_h_check)
        
        # Flip vertical
        self.flip_v_check = QCheckBox("Flip vertically")
        video_layout.addRow("", self.flip_v_check)
        
        # Color mode
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["RGB", "Grayscale"])
        video_layout.addRow("Color Mode:", self.color_mode_combo)
        
        layout.addWidget(video_group)
        
        # Add stretch to push content to the top
        layout.addStretch()
        
        return tab
    
    def _create_detector_tab(self):
        """Create detector settings tab.
        
        Returns:
            QWidget: The detector settings widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Face detector group
        face_group = QGroupBox("Face Detector")
        face_layout = QFormLayout(face_group)
        
        # Enable face detector
        self.face_detector_check = QCheckBox("Enable Face Detector")
        face_layout.addRow("", self.face_detector_check)
        
        # Face detection confidence
        self.face_confidence_spin = QDoubleSpinBox()
        self.face_confidence_spin.setRange(0.1, 1.0)
        self.face_confidence_spin.setSingleStep(0.05)
        self.face_confidence_spin.setDecimals(2)
        face_layout.addRow("Detection Confidence:", self.face_confidence_spin)
        
        # Face model selection
        self.face_model_combo = QComboBox()
        self.face_model_combo.addItem("Short Range (2m)", 0)
        self.face_model_combo.addItem("Full Range (5m)", 1)
        face_layout.addRow("Model Selection:", self.face_model_combo)
        
        layout.addWidget(face_group)
        
        # Expression detector group
        expr_group = QGroupBox("Expression Detector")
        expr_layout = QFormLayout(expr_group)
        
        # Enable expression detector
        self.expr_detector_check = QCheckBox("Enable Expression Detector")
        expr_layout.addRow("", self.expr_detector_check)
        
        # Expression detection confidence
        self.expr_confidence_spin = QDoubleSpinBox()
        self.expr_confidence_spin.setRange(0.1, 1.0)
        self.expr_confidence_spin.setSingleStep(0.05)
        self.expr_confidence_spin.setDecimals(2)
        expr_layout.addRow("Detection Confidence:", self.expr_confidence_spin)
        
        # Expression tracking confidence
        self.expr_tracking_spin = QDoubleSpinBox()
        self.expr_tracking_spin.setRange(0.1, 1.0)
        self.expr_tracking_spin.setSingleStep(0.05)
        self.expr_tracking_spin.setDecimals(2)
        expr_layout.addRow("Tracking Confidence:", self.expr_tracking_spin)
        
        # Max faces
        self.expr_max_faces_spin = QSpinBox()
        self.expr_max_faces_spin.setRange(1, 10)
        expr_layout.addRow("Maximum Faces:", self.expr_max_faces_spin)
        
        # Enabled expressions
        self.expr_smile_check = QCheckBox("Smile")
        self.expr_frown_check = QCheckBox("Frown")
        self.expr_surprise_check = QCheckBox("Surprise")
        self.expr_neutral_check = QCheckBox("Neutral")
        
        expr_layout.addRow("Enabled Expressions:", self.expr_smile_check)
        expr_layout.addRow("", self.expr_frown_check)
        expr_layout.addRow("", self.expr_surprise_check)
        expr_layout.addRow("", self.expr_neutral_check)
        
        layout.addWidget(expr_group)
        
        # Add stretch to push content to the top
        layout.addStretch()
        
        return tab
    
    def _create_analytics_tab(self):
        """Create analytics settings tab.
        
        Returns:
            QWidget: The analytics settings widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analytics collection group
        analytics_group = QGroupBox("Analytics Collection")
        analytics_layout = QFormLayout(analytics_group)
        
        # Enable analytics
        self.analytics_enabled_check = QCheckBox("Enable Analytics Collection")
        analytics_layout.addRow("", self.analytics_enabled_check)
        
        # Storage path
        self.analytics_path_edit = QLineEdit()
        self.analytics_path_edit.setReadOnly(True)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.analytics_path_edit)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_analytics_path)
        path_layout.addWidget(browse_button)
        
        analytics_layout.addRow("Storage Path:", path_layout)
        
        layout.addWidget(analytics_group)
        
        # Session recording group
        session_group = QGroupBox("Session Recording")
        session_layout = QFormLayout(session_group)
        
        # Enable session recording
        self.session_recording_check = QCheckBox("Enable Session Recording")
        session_layout.addRow("", self.session_recording_check)
        
        # Recording format
        self.recording_format_combo = QComboBox()
        self.recording_format_combo.addItems(["JSON", "CSV", "SQLite"])
        session_layout.addRow("Recording Format:", self.recording_format_combo)
        
        layout.addWidget(session_group)
        
        # Data retention group
        retention_group = QGroupBox("Data Retention")
        retention_layout = QFormLayout(retention_group)
        
        # Data retention period
        self.retention_combo = QComboBox()
        self.retention_combo.addItems(["1 day", "1 week", "1 month", "3 months", "6 months", "1 year", "Forever"])
        retention_layout.addRow("Retain Data For:", self.retention_combo)
        
        # Auto-cleanup
        self.auto_cleanup_check = QCheckBox("Automatically clean up old data")
        retention_layout.addRow("", self.auto_cleanup_check)
        
        layout.addWidget(retention_group)
        
        # Add stretch to push content to the top
        layout.addStretch()
        
        return tab
    
    def _create_advanced_tab(self):
        """Create advanced settings tab.
        
        Returns:
            QWidget: The advanced settings widget
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Logging group
        logging_group = QGroupBox("Logging")
        logging_layout = QFormLayout(logging_group)
        
        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["Debug", "Info", "Warning", "Error", "Critical"])
        logging_layout.addRow("Log Level:", self.log_level_combo)
        
        # File logging
        self.file_logging_check = QCheckBox("Enable File Logging")
        logging_layout.addRow("", self.file_logging_check)
        
        # Log directory
        self.log_dir_edit = QLineEdit()
        self.log_dir_edit.setReadOnly(True)
        
        log_dir_layout = QHBoxLayout()
        log_dir_layout.addWidget(self.log_dir_edit)
        
        log_browse_button = QPushButton("Browse...")
        log_browse_button.clicked.connect(self._browse_log_directory)
        log_dir_layout.addWidget(log_browse_button)
        
        logging_layout.addRow("Log Directory:", log_dir_layout)
        
        layout.addWidget(logging_group)
        
        # Performance group
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        # Thread count
        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, 16)
        perf_layout.addRow("Worker Threads:", self.thread_count_spin)
        
        # GPU acceleration
        self.gpu_accel_check = QCheckBox("Enable GPU Acceleration (if available)")
        perf_layout.addRow("", self.gpu_accel_check)
        
        # Frame skipping
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(0, 5)
        self.frame_skip_spin.setSpecialValueText("None")
        perf_layout.addRow("Frame Skipping:", self.frame_skip_spin)
        
        layout.addWidget(perf_group)
        
        # Developer group
        dev_group = QGroupBox("Developer Options")
        dev_layout = QFormLayout(dev_group)
        
        # Debug mode
        self.debug_mode_check = QCheckBox("Enable Debug Mode")
        dev_layout.addRow("", self.debug_mode_check)
        
        # Performance monitoring
        self.perf_monitoring_check = QCheckBox("Enable Performance Monitoring")
        dev_layout.addRow("", self.perf_monitoring_check)
        
        layout.addWidget(dev_group)
        
        # Add stretch to push content to the top
        layout.addStretch()
        
        return tab
    
    def _load_settings(self):
        """Load current settings into UI controls."""
        # General settings
        ui_theme = self.config.get("ui", "theme", "default")
        theme_index = self.theme_combo.findText(ui_theme.capitalize())
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)
        
        # Language defaults to English
        language = self.config.get("ui", "language", "English")
        language_index = self.language_combo.findText(language)
        if language_index >= 0:
            self.language_combo.setCurrentIndex(language_index)
        
        # Application behavior
        self.start_camera_check.setChecked(self.config.get("camera", "auto_start", True))
        self.auto_save_check.setChecked(self.config.get("general", "auto_save", True))
        
        # Camera settings
        camera_id = self.config.get("camera", "camera_id", 0)
        camera_index = self.camera_source_combo.findData(camera_id)
        if camera_index >= 0:
            self.camera_source_combo.setCurrentIndex(camera_index)
        
        resolution = self.config.get("camera", "resolution", (640, 480))
        resolution_text = f"{resolution[0]} x {resolution[1]}"
        resolution_index = -1
        for i in range(self.resolution_combo.count()):
            if self.resolution_combo.itemText(i) == resolution_text:
                resolution_index = i
                break
        if resolution_index >= 0:
            self.resolution_combo.setCurrentIndex(resolution_index)
        
        self.fps_spin.setValue(self.config.get("camera", "fps", 30))
        self.flip_h_check.setChecked(self.config.get("camera", "flip_h", False))
        self.flip_v_check.setChecked(self.config.get("camera", "flip_v", False))
        
        color_mode = self.config.get("camera", "color_mode", "RGB")
        color_index = self.color_mode_combo.findText(color_mode)
        if color_index >= 0:
            self.color_mode_combo.setCurrentIndex(color_index)
        
        # Detector settings
        face_detector_config = self.config.get_detector_config("face")
        self.face_detector_check.setChecked(face_detector_config.get("enabled", True))
        self.face_confidence_spin.setValue(face_detector_config.get("min_detection_confidence", 0.5))
        
        face_model = face_detector_config.get("model_selection", 0)
        face_model_index = self.face_model_combo.findData(face_model)
        if face_model_index >= 0:
            self.face_model_combo.setCurrentIndex(face_model_index)
        
        expr_detector_config = self.config.get_detector_config("expression")
        self.expr_detector_check.setChecked(expr_detector_config.get("enabled", True))
        self.expr_confidence_spin.setValue(expr_detector_config.get("min_detection_confidence", 0.5))
        self.expr_tracking_spin.setValue(expr_detector_config.get("min_tracking_confidence", 0.5))
        self.expr_max_faces_spin.setValue(expr_detector_config.get("max_num_faces", 1))
        
        enabled_expressions = expr_detector_config.get("enable_expressions", [])
        self.expr_smile_check.setChecked("smile" in enabled_expressions)
        self.expr_frown_check.setChecked("frown" in enabled_expressions)
        self.expr_surprise_check.setChecked("surprise" in enabled_expressions)
        self.expr_neutral_check.setChecked("neutral" in enabled_expressions)
        
        # Analytics settings
        analytics_config = self.config.get("analytics") or {}
        self.analytics_enabled_check.setChecked(analytics_config.get("enabled", True))
        self.analytics_path_edit.setText(analytics_config.get("storage_path", "../data/analytics"))
        self.session_recording_check.setChecked(analytics_config.get("session_recording", False))
        
        recording_format = analytics_config.get("recording_format", "JSON")
        format_index = self.recording_format_combo.findText(recording_format)
        if format_index >= 0:
            self.recording_format_combo.setCurrentIndex(format_index)
        
        retention_period = analytics_config.get("retention_period", "1 month")
        retention_index = self.retention_combo.findText(retention_period)
        if retention_index >= 0:
            self.retention_combo.setCurrentIndex(retention_index)
        
        self.auto_cleanup_check.setChecked(analytics_config.get("auto_cleanup", True))
        
        # Advanced settings
        logging_config = self.config.get("logging") or {}
        log_level = logging_config.get("level", "info").capitalize()
        level_index = self.log_level_combo.findText(log_level)
        if level_index >= 0:
            self.log_level_combo.setCurrentIndex(level_index)
        
        self.file_logging_check.setChecked(logging_config.get("file_logging", True))
        self.log_dir_edit.setText(logging_config.get("log_directory", "../logs"))
        
        performance_config = self.config.get("performance") or {}
        self.thread_count_spin.setValue(performance_config.get("worker_threads", 4))
        self.gpu_accel_check.setChecked(performance_config.get("gpu_acceleration", True))
        self.frame_skip_spin.setValue(performance_config.get("frame_skip", 0))
        
        developer_config = self.config.get("developer") or {}
        self.debug_mode_check.setChecked(developer_config.get("debug_mode", False))
        self.perf_monitoring_check.setChecked(developer_config.get("performance_monitoring", False))
    
    def _save_settings(self):
        """Save settings from UI controls to configuration."""
        # General settings
        self.config.set("ui", "theme", self.theme_combo.currentText().lower())
        self.config.set("ui", "language", self.language_combo.currentText())
        
        # Application behavior
        self.config.set("camera", "auto_start", self.start_camera_check.isChecked())
        self.config.set("general", "auto_save", self.auto_save_check.isChecked())
        
        # Camera settings
        self.config.set("camera", "camera_id", self.camera_source_combo.currentData())
        self.config.set("camera", "resolution", self.resolution_combo.currentData())
        self.config.set("camera", "fps", self.fps_spin.value())
        self.config.set("camera", "flip_h", self.flip_h_check.isChecked())
        self.config.set("camera", "flip_v", self.flip_v_check.isChecked())
        self.config.set("camera", "color_mode", self.color_mode_combo.currentText())
        
        # Detector settings
        face_detector_config = {
            "enabled": self.face_detector_check.isChecked(),
            "min_detection_confidence": self.face_confidence_spin.value(),
            "model_selection": self.face_model_combo.currentData()
        }
        self.config.set_detector_config("face", face_detector_config)
        
        # Build list of enabled expressions
        enabled_expressions = []
        if self.expr_smile_check.isChecked():
            enabled_expressions.append("smile")
        if self.expr_frown_check.isChecked():
            enabled_expressions.append("frown")
        if self.expr_surprise_check.isChecked():
            enabled_expressions.append("surprise")
        if self.expr_neutral_check.isChecked():
            enabled_expressions.append("neutral")
        
        expr_detector_config = {
            "enabled": self.expr_detector_check.isChecked(),
            "min_detection_confidence": self.expr_confidence_spin.value(),
            "min_tracking_confidence": self.expr_tracking_spin.value(),
            "max_num_faces": self.expr_max_faces_spin.value(),
            "enable_expressions": enabled_expressions
        }
        self.config.set_detector_config("expression", expr_detector_config)
        
        # Analytics settings
        analytics_config = {
            "enabled": self.analytics_enabled_check.isChecked(),
            "storage_path": self.analytics_path_edit.text(),
            "session_recording": self.session_recording_check.isChecked(),
            "recording_format": self.recording_format_combo.currentText(),
            "retention_period": self.retention_combo.currentText(),
            "auto_cleanup": self.auto_cleanup_check.isChecked()
        }
        self.config.config["analytics"] = analytics_config
        
        # Advanced settings
        logging_config = {
            "level": self.log_level_combo.currentText().lower(),
            "file_logging": self.file_logging_check.isChecked(),
            "log_directory": self.log_dir_edit.text()
        }
        self.config.config["logging"] = logging_config
        
        performance_config = {
            "worker_threads": self.thread_count_spin.value(),
            "gpu_acceleration": self.gpu_accel_check.isChecked(),
            "frame_skip": self.frame_skip_spin.value()
        }
        self.config.config["performance"] = performance_config
        
        developer_config = {
            "debug_mode": self.debug_mode_check.isChecked(),
            "performance_monitoring": self.perf_monitoring_check.isChecked()
        }
        self.config.config["developer"] = developer_config
        
        # Save to file
        self.config.save()
        
        # Show confirmation message
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
    
    def _reset_settings(self):
        """Reset settings to defaults."""
        # Confirm with user
        response = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            # Reset to defaults
            self.config.load_defaults()
            
            # Reload UI
            self._load_settings()
            
            # Show confirmation message
            QMessageBox.information(self, "Settings Reset", "Settings have been reset to defaults.")
    
    def _browse_analytics_path(self):
        """Browse for analytics storage directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Analytics Storage Directory", self.analytics_path_edit.text()
        )
        
        if directory:
            self.analytics_path_edit.setText(directory)
    
    def _browse_log_directory(self):
        """Browse for log directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Log Directory", self.log_dir_edit.text()
        )
        
        if directory:
            self.log_dir_edit.setText(directory)
    
    def update_tab(self):
        """Update the tab content.
        
        This method is called regularly from the main window.
        """
        # Nothing to update at the moment
        pass
    
    def on_tab_selected(self):
        """Handle tab selected event.
        
        This method is called when this tab is selected in the UI.
        """
        # Reload settings when tab is selected
        self._load_settings()