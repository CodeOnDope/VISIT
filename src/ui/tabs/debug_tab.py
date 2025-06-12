"""
Debug tab for the VISIT Museum Tracker application.

This module implements the debug tab UI that shows logs, performance metrics,
and debugging information for the application.
"""

import os
import logging
import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTextEdit, QTabWidget, QCheckBox,
    QFormLayout, QSpinBox, QFileDialog, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

from src.utils.logger import setup_logger


class DebugTab(QWidget):
    """Tab for debugging and monitoring the application."""
    
    def __init__(self, logger, parent=None):
        """Initialize the debug tab.
        
        Args:
            logger (logging.Logger): Application logger
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Store logger reference
        self.logger = logger
        
        # Set up UI
        self._init_ui()
        
        # Set up update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_performance_stats)
        self.update_timer.start(1000)  # Update every 1 second
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create debug tabs
        debug_tabs = QTabWidget()
        
        # Logs tab
        logs_widget = self._create_logs_tab()
        debug_tabs.addTab(logs_widget, "Logs")
        
        # Performance tab
        performance_widget = self._create_performance_tab()
        debug_tabs.addTab(performance_widget, "Performance")
        
        # Components tab
        components_widget = self._create_components_tab()
        debug_tabs.addTab(components_widget, "Components")
        
        # Configuration tab
        config_widget = self._create_config_tab()
        debug_tabs.addTab(config_widget, "Configuration")
        
        # Add tabs to main layout
        main_layout.addWidget(debug_tabs)
    
    def _create_logs_tab(self):
        """Create logs tab.
        
        Returns:
            QWidget: The logs tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Log level filter
        controls_layout.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        self.log_level_combo.currentTextChanged.connect(self._update_log_filter)
        controls_layout.addWidget(self.log_level_combo)
        
        # Log source filter
        controls_layout.addWidget(QLabel("Source:"))
        self.log_source_combo = QComboBox()
        self.log_source_combo.addItems(["All", "Application", "Detectors", "Camera", "UI"])
        self.log_source_combo.currentTextChanged.connect(self._update_log_filter)
        controls_layout.addWidget(self.log_source_combo)
        
        # Auto-scroll checkbox
        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_check)
        
        # Clear button
        self.clear_log_button = QPushButton("Clear Logs")
        self.clear_log_button.clicked.connect(self._clear_logs)
        controls_layout.addWidget(self.clear_log_button)
        
        # Export button
        self.export_log_button = QPushButton("Export Logs")
        self.export_log_button.clicked.connect(self._export_logs)
        controls_layout.addWidget(self.export_log_button)
        
        layout.addLayout(controls_layout)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_display)
        
        # Add some sample logs
        self._add_sample_logs()
        
        return widget
    
    def _create_performance_tab(self):
        """Create performance tab.
        
        Returns:
            QWidget: The performance tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Vertical)
        
        # Top panel - Performance metrics
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        
        # Real-time metrics group
        metrics_group = QGroupBox("Real-time Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        # FPS
        self.fps_label = QLabel("0")
        metrics_layout.addRow("FPS:", self.fps_label)
        
        # CPU usage
        self.cpu_label = QLabel("0%")
        metrics_layout.addRow("CPU Usage:", self.cpu_label)
        
        # Memory usage
        self.memory_label = QLabel("0 MB")
        metrics_layout.addRow("Memory Usage:", self.memory_label)
        
        # Processing latency
        self.latency_label = QLabel("0 ms")
        metrics_layout.addRow("Processing Latency:", self.latency_label)
        
        top_layout.addWidget(metrics_group)
        
        # Detector performance group
        detector_group = QGroupBox("Detector Performance")
        detector_layout = QFormLayout(detector_group)
        
        # Face detector
        self.face_detector_label = QLabel("0 ms")
        detector_layout.addRow("Face Detector:", self.face_detector_label)
        
        # Expression detector
        self.expression_detector_label = QLabel("0 ms")
        detector_layout.addRow("Expression Detector:", self.expression_detector_label)
        
        top_layout.addWidget(detector_group)
        
        # Bottom panel - Performance history
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        
        # Performance history table
        self.performance_table = QTableWidget(0, 5)
        self.performance_table.setHorizontalHeaderLabels(
            ["Time", "FPS", "CPU Usage", "Memory", "Latency"]
        )
        self.performance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bottom_layout.addWidget(self.performance_table)
        
        # Add panels to splitter
        splitter.addWidget(top_panel)
        splitter.addWidget(bottom_panel)
        
        # Set initial sizes (40% metrics, 60% history)
        splitter.setSizes([400, 600])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        return widget
    
    def _create_components_tab(self):
        """Create components tab.
        
        Returns:
            QWidget: The components tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Components table
        self.components_table = QTableWidget(0, 4)
        self.components_table.setHorizontalHeaderLabels(
            ["Component", "Status", "Version", "Details"]
        )
        self.components_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.components_table)
        
        # Add sample component data
        self._add_sample_components()
        
        return widget
    
    def _create_config_tab(self):
        """Create configuration tab.
        
        Returns:
            QWidget: The configuration tab widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Debug options group
        debug_group = QGroupBox("Debug Options")
        debug_layout = QVBoxLayout(debug_group)
        
        # Debug checkboxes
        self.show_fps_check = QCheckBox("Show FPS in Overlays")
        self.show_fps_check.setChecked(True)
        debug_layout.addWidget(self.show_fps_check)
        
        self.show_detection_time_check = QCheckBox("Show Detection Processing Time")
        self.show_detection_time_check.setChecked(True)
        debug_layout.addWidget(self.show_detection_time_check)
        
        self.log_detections_check = QCheckBox("Log Detections")
        debug_layout.addWidget(self.log_detections_check)
        
        self.debug_mode_check = QCheckBox("Enable Debug Mode")
        debug_layout.addWidget(self.debug_mode_check)
        
        layout.addWidget(debug_group)
        
        # Log settings group
        log_group = QGroupBox("Log Settings")
        log_layout = QFormLayout(log_group)
        
        # Log level
        self.log_level_spin = QComboBox()
        self.log_level_spin.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_spin.setCurrentText("INFO")
        log_layout.addRow("Log Level:", self.log_level_spin)
        
        # Max log file size
        self.log_size_spin = QSpinBox()
        self.log_size_spin.setRange(1, 100)
        self.log_size_spin.setValue(10)
        self.log_size_spin.setSuffix(" MB")
        log_layout.addRow("Max Log File Size:", self.log_size_spin)
        
        # Log file count
        self.log_count_spin = QSpinBox()
        self.log_count_spin.setRange(1, 20)
        self.log_count_spin.setValue(5)
        log_layout.addRow("Max Log Files:", self.log_count_spin)
        
        layout.addWidget(log_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        
        # Action buttons
        self.clear_cache_button = QPushButton("Clear Cache")
        self.clear_cache_button.clicked.connect(self._clear_cache)
        actions_layout.addWidget(self.clear_cache_button)
        
        self.reset_logs_button = QPushButton("Reset Logs")
        self.reset_logs_button.clicked.connect(self._reset_logs)
        actions_layout.addWidget(self.reset_logs_button)
        
        self.apply_debug_settings_button = QPushButton("Apply Settings")
        self.apply_debug_settings_button.clicked.connect(self._apply_debug_settings)
        actions_layout.addWidget(self.apply_debug_settings_button)
        
        layout.addWidget(actions_group)
        
        # Add stretch to push content to the top
        layout.addStretch()
        
        return widget
    
    def _add_sample_logs(self):
        """Add sample log entries for demonstration."""
        self.log_display.append("<font color='blue'>[INFO] [2025-05-11 10:00:12] [MainWindow] VISIT Museum Tracker application started</font>")
        self.log_display.append("<font color='blue'>[INFO] [2025-05-11 10:00:13] [Camera] Camera initialized with ID: 0</font>")
        self.log_display.append("<font color='blue'>[INFO] [2025-05-11 10:00:14] [DetectorManager] Loading detectors...</font>")
        self.log_display.append("<font color='blue'>[INFO] [2025-05-11 10:00:15] [FaceDetector] Face detector initialized</font>")
        self.log_display.append("<font color='blue'>[INFO] [2025-05-11 10:00:16] [ExpressionDetector] Expression detector initialized</font>")
        self.log_display.append("<font color='green'>[DEBUG] [2025-05-11 10:00:17] [FaceDetector] Setting min_detection_confidence to 0.5</font>")
        self.log_display.append("<font color='orange'>[WARNING] [2025-05-11 10:00:18] [Camera] Camera resolution adjusted to 640x480</font>")
    
    def _add_sample_components(self):
        """Add sample component data for demonstration."""
        self.components_table.setRowCount(8)
        
        # MediaPipe components
        self._add_component(0, "MediaPipe", "Running", "0.8.9", "Face detection module loaded")
        self._add_component(1, "MediaPipe Face Mesh", "Running", "0.8.9", "368 landmarks")
        
        # OpenCV components
        self._add_component(2, "OpenCV", "Running", "4.5.3", "With CUDA support")
        self._add_component(3, "Camera", "Running", "1.0", "640x480 @ 30 FPS")
        
        # Application components
        self._add_component(4, "Face Detector", "Running", "1.0", "Min confidence: 0.5")
        self._add_component(5, "Expression Detector", "Running", "1.0", "Min confidence: 0.5")
        self._add_component(6, "Analytics Engine", "Running", "1.0", "SQLite backend")
        self._add_component(7, "UI", "Running", "1.0", "PyQt5 5.15.2")
    
    def _add_component(self, row, name, status, version, details):
        """Add a component to the components table.
        
        Args:
            row (int): Row index
            name (str): Component name
            status (str): Component status
            version (str): Component version
            details (str): Component details
        """
        self.components_table.setItem(row, 0, QTableWidgetItem(name))
        status_item = QTableWidgetItem(status)
        status_item.setForeground(QColor("green") if status == "Running" else QColor("red"))
        self.components_table.setItem(row, 1, status_item)
        self.components_table.setItem(row, 2, QTableWidgetItem(version))
        self.components_table.setItem(row, 3, QTableWidgetItem(details))
    
    def _update_log_filter(self):
        """Update log display based on current filter settings."""
        # Placeholder implementation
        self.logger.info(f"Log filter updated: Level={self.log_level_combo.currentText()}, Source={self.log_source_combo.currentText()}")
    
    def _clear_logs(self):
        """Clear log display."""
        self.log_display.clear()
        self.logger.info("Logs cleared")
    
    def _export_logs(self):
        """Export logs to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "", "Log Files (*.log);;Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                f.write(self.log_display.toPlainText())
            
            self.logger.info(f"Logs exported to: {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        # Simulate changing values
        import random
        
        # Update current metrics
        fps = random.uniform(25.0, 30.0)
        cpu = random.uniform(10.0, 30.0)
        memory = random.uniform(50.0, 150.0)
        latency = random.uniform(10.0, 50.0)
        
        self.fps_label.setText(f"{fps:.1f}")
        self.cpu_label.setText(f"{cpu:.1f}%")
        self.memory_label.setText(f"{memory:.1f} MB")
        self.latency_label.setText(f"{latency:.1f} ms")
        
        # Update detector performance
        face_time = random.uniform(5.0, 15.0)
        expr_time = random.uniform(10.0, 25.0)
        
        self.face_detector_label.setText(f"{face_time:.1f} ms")
        self.expression_detector_label.setText(f"{expr_time:.1f} ms")
        
        # Add to history table (limit to 100 rows)
        if self.performance_table.rowCount() >= 100:
            self.performance_table.removeRow(0)
        
        row = self.performance_table.rowCount()
        self.performance_table.insertRow(row)
        
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.performance_table.setItem(row, 0, QTableWidgetItem(time_str))
        self.performance_table.setItem(row, 1, QTableWidgetItem(f"{fps:.1f}"))
        self.performance_table.setItem(row, 2, QTableWidgetItem(f"{cpu:.1f}%"))
        self.performance_table.setItem(row, 3, QTableWidgetItem(f"{memory:.1f} MB"))
        self.performance_table.setItem(row, 4, QTableWidgetItem(f"{latency:.1f} ms"))
        
        # Scroll to bottom if auto-scroll is enabled
        if self.auto_scroll_check.isChecked():
            self.performance_table.scrollToBottom()
    
    def _clear_cache(self):
        """Clear application cache."""
        # Placeholder implementation
        self.logger.info("Cache cleared")
    
    def _reset_logs(self):
        """Reset log files."""
        # Placeholder implementation
        self.logger.info("Log files reset")
    
    def _apply_debug_settings(self):
        """Apply debug settings."""
        # Placeholder implementation
        self.logger.info("Debug settings applied")
    
    def update_tab(self):
        """Update the tab content.
        
        This method is called regularly from the main window.
        """
        # Performance updates are handled by timer
        pass
    
    def on_tab_selected(self):
        """Handle tab selected event.
        
        This method is called when this tab is selected in the UI.
        """
        # Start performance updates
        if not self.update_timer.isActive():
            self.update_timer.start(1000)
    
    def _log_message(self, level, message):
        """Add a log message to the log display.
        
        Args:
            level (str): Log level
            message (str): Log message
        """
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Color based on level
        color = {
            "DEBUG": "green",
            "INFO": "blue",
            "WARNING": "orange",
            "ERROR": "red",
            "CRITICAL": "purple"
        }.get(level, "black")
        
        # Format log entry
        log_entry = f"<font color='{color}'>[{level}] [{time_str}] [DebugTab] {message}</font>"
        
        # Add to display
        self.log_display.append(log_entry)
        
        # Scroll to bottom if auto-scroll is enabled
        if self.auto_scroll_check.isChecked():
            self.log_display.verticalScrollBar().setValue(
                self.log_display.verticalScrollBar().maximum()
            )
