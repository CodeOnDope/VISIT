"""
Video player component for the VISIT Museum Tracker system.

This module implements a video player widget that displays video frames from
either a camera or a video file, with optional detection overlays.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class VideoPlayer(QWidget):
    """Video player widget for displaying camera feeds or video files."""
    
    # Signals
    frame_updated = pyqtSignal(np.ndarray)
    mouse_clicked = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)
    mouse_double_clicked = pyqtSignal(int, int)
    key_pressed = pyqtSignal(int)
    
    def __init__(self, camera, parent=None):
        """Initialize the video player.
        
        Args:
            camera (Camera): Camera instance for video capture
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Store camera reference
        self.camera = camera
        
        # Initialize state
        self.current_frame = None
        self.overlay_enabled = True
        self.resizing_mode = "contain" # "contain", "stretch", "crop"
        
        # Zone editor
        self.zone_editor = None
        self.draw_points = []
        
        # Set up UI
        self._init_ui()
        
        # Set up timer for frame updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_frame)
        self.update_timer.start(33)  # ~30 FPS
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create display label
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setStyleSheet("background-color: black;")
        
        # Add display to layout
        layout.addWidget(self.display_label)
        
        # Create controls layout
        controls_layout = QHBoxLayout()
        
        # Overlay toggle
        self.overlay_checkbox = QCheckBox("Show Overlays")
        self.overlay_checkbox.setChecked(self.overlay_enabled)
        self.overlay_checkbox.toggled.connect(self._toggle_overlay)
        controls_layout.addWidget(self.overlay_checkbox)
        
        # Resize mode selector
        controls_layout.addWidget(QLabel("Resize:"))
        self.resize_combo = QComboBox()
        self.resize_combo.addItems(["Contain", "Stretch", "Crop"])
        self.resize_combo.setCurrentText("Contain")
        self.resize_combo.currentTextChanged.connect(self._change_resize_mode)
        controls_layout.addWidget(self.resize_combo)
        
        # Zoom control
        controls_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(100, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._update_frame)
        controls_layout.addWidget(self.zoom_slider)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
    
    def _toggle_overlay(self, enabled):
        """Toggle detection overlay display.
        
        Args:
            enabled (bool): Whether overlay should be enabled
        """
        self.overlay_enabled = enabled
    
    def _change_resize_mode(self, mode):
        """Change the frame resizing mode.
        
        Args:
            mode (str): The resize mode ("Contain", "Stretch", or "Crop")
        """
        self.resizing_mode = mode.lower()
        self._update_frame()
    
    def _update_frame(self):
        """Update the video display with the latest frame."""
        if not self.camera.is_running:
            return
        
        # Get frame from camera
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        # Store current frame
        self.current_frame = frame.copy()
        
        # Emit signal with the raw frame for detectors
        self.frame_updated.emit(frame)
        
        # Apply any processing
        display_frame = self._process_frame(frame)
        
        # Convert to QImage and display
        self._display_frame(display_frame)
    
    def _process_frame(self, frame):
        """Process a video frame before display.
        
        Args:
            frame (numpy.ndarray): The raw video frame
            
        Returns:
            numpy.ndarray: The processed frame
        """
        # Currently no processing is done, just return the frame
        # Overlay is applied by the detector and already in the frame
        return frame
    
    def _display_frame(self, frame):
        """Display a frame in the video player.
        
        Args:
            frame (numpy.ndarray): The frame to display
        """
        if frame is None:
            return
        
        # Get dimensions
        h, w, ch = frame.shape
        display_w = self.display_label.width()
        display_h = self.display_label.height()
        
        # Skip if dimensions are invalid
        if display_w <= 0 or display_h <= 0:
            return
        
        # Get zoom factor
        zoom = self.zoom_slider.value() / 100.0
        
        # Calculate target size based on resize mode
        if self.resizing_mode == "contain":
            # Maintain aspect ratio and fit within display
            scale = min(display_w / w, display_h / h)
            target_w = int(w * scale * zoom)
            target_h = int(h * scale * zoom)
            
        elif self.resizing_mode == "stretch":
            # Stretch to fill display area
            target_w = display_w
            target_h = display_h
            
        elif self.resizing_mode == "crop":
            # Maintain aspect ratio and crop to fill display
            scale = max(display_w / w, display_h / h)
            target_w = int(w * scale * zoom)
            target_h = int(h * scale * zoom)
        
        # Resize the frame
        if target_w > 0 and target_h > 0:
            frame_resized = cv2.resize(frame, (target_w, target_h))
            
            # Convert BGR to RGB for QImage
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Create QImage from the frame
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], 
                         frame_rgb.strides[0], QImage.Format_RGB888)
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(image)
            self.display_label.setPixmap(pixmap)
            
            # Update for zone editor drawing
            self.update()
    
    def show_frame(self, frame):
        """Show a specific frame in the video player.
        
        This can be used to display processed frames from detectors.
        
        Args:
            frame (numpy.ndarray): The frame to display
        """
        if self.overlay_enabled and frame is not None:
            self._display_frame(frame)
    
    def get_current_frame(self):
        """Get the current frame.
        
        Returns:
            numpy.ndarray: The current video frame
        """
        return self.current_frame
    
    def set_zone_editor(self, editor):
        """Set the zone editor for handling mouse events.
        
        Args:
            editor (ZoneEditor): The zone editor instance
        """
        self.zone_editor = editor
        
        # Ensure key events work by setting focus policy
        self.setFocusPolicy(Qt.StrongFocus)
    
    def mousePressEvent(self, event):
        """Handle mouse press events.
        
        Args:
            event (QMouseEvent): The mouse event
        """
        super().mousePressEvent(event)
        
        # Convert to image coordinates
        x, y = self._get_image_coordinates(event.x(), event.y())
        if x is not None and y is not None:
            # Emit signal
            self.mouse_clicked.emit(x, y)
            
            # Handle in zone editor
            if self.zone_editor and self.zone_editor.is_editing():
                self.zone_editor.handle_mouse_event("press", x, y)
                # Force repaint to update zone drawing
                self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events.
        
        Args:
            event (QMouseEvent): The mouse event
        """
        super().mouseMoveEvent(event)
        
        # Convert to image coordinates
        x, y = self._get_image_coordinates(event.x(), event.y())
        if x is not None and y is not None:
            # Emit signal
            self.mouse_moved.emit(x, y)
            
            # Handle in zone editor
            if self.zone_editor and self.zone_editor.is_editing():
                self.zone_editor.handle_mouse_event("move", x, y)
                # Force repaint to update zone drawing
                self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events.
        
        Args:
            event (QMouseEvent): The mouse event
        """
        super().mouseReleaseEvent(event)
        
        # Convert to image coordinates
        x, y = self._get_image_coordinates(event.x(), event.y())
        if x is not None and y is not None:
            # Emit signal
            self.mouse_released.emit(x, y)
            
            # Handle in zone editor
            if self.zone_editor and self.zone_editor.is_editing():
                self.zone_editor.handle_mouse_event("release", x, y)
                # Force repaint to update zone drawing
                self.update()

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double click events.
        
        Args:
            event (QMouseEvent): The mouse event
        """
        super().mouseDoubleClickEvent(event)
        
        # Convert to image coordinates
        x, y = self._get_image_coordinates(event.x(), event.y())
        if x is not None and y is not None:
            # Emit signal
            self.mouse_double_clicked.emit(x, y)
            
            # Handle in zone editor
            if self.zone_editor and self.zone_editor.is_editing():
                self.zone_editor.handle_mouse_event("double_click", x, y)
                # Force repaint to update zone drawing
                self.update()

    def keyPressEvent(self, event):
        """Handle key press events.
        
        Args:
            event (QKeyEvent): The key event
        """
        super().keyPressEvent(event)
        
        # Emit signal
        self.key_pressed.emit(event.key())
        
        # Handle in zone editor
        if self.zone_editor and self.zone_editor.is_editing():
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.zone_editor.handle_mouse_event("enter_key", 0, 0)
                # Force repaint to update zone drawing
                self.update()
            elif event.key() == Qt.Key_Escape:
                self.zone_editor.handle_mouse_event("escape_key", 0, 0)
                # Force repaint to update zone drawing
                self.update()

    def _get_image_coordinates(self, widget_x, widget_y):
        """Convert widget coordinates to image coordinates.
        
        Args:
            widget_x (int): X coordinate in widget space
            widget_y (int): Y coordinate in widget space
            
        Returns:
            tuple: (x, y) coordinates in image space, or (None, None) if outside image
        """
        if self.current_frame is None:
            return None, None
        
        # Get display label geometry
        label_rect = self.display_label.rect()
        pixmap = self.display_label.pixmap()
        
        if not pixmap:
            return None, None
        
        # Get the coordinates within the label
        x_rel = widget_x - self.display_label.x()
        y_rel = widget_y - self.display_label.y()
        
        # Calculate the image position and size within the label
        pixmap_rect = pixmap.rect()
        pixmap_size = pixmap.size()
        
        # Get the position of the image within the label
        x_offset = (label_rect.width() - pixmap_size.width()) / 2
        y_offset = (label_rect.height() - pixmap_size.height()) / 2
        
        # Calculate image coordinates
        image_x = int((x_rel - x_offset) * (self.current_frame.shape[1] / pixmap_size.width()))
        image_y = int((y_rel - y_offset) * (self.current_frame.shape[0] / pixmap_size.height()))
        
        # Check if within image bounds
        if (0 <= image_x < self.current_frame.shape[1] and
            0 <= image_y < self.current_frame.shape[0]):
            return image_x, image_y
        
        return None, None

    def paintEvent(self, event):
        """Override paint event to draw zone editing overlay.
        
        Args:
            event (QPaintEvent): The paint event
        """
        super().paintEvent(event)
        
        # If zone editor is active and editing, draw the current points
        if self.zone_editor and self.zone_editor.is_editing():
            painter = QPainter(self)
            
            # Draw current points and lines
            points = self.zone_editor.get_current_points()
            
            if points:
                # Convert to widget coordinates
                widget_points = []
                for point in points:
                    wx, wy = self._get_widget_coordinates(point[0], point[1])
                    if wx is not None and wy is not None:
                        widget_points.append((wx, wy))
                
                # Draw points and lines
                if widget_points:
                    pen = QPen(QColor(255, 0, 0))
                    pen.setWidth(3)
                    painter.setPen(pen)
                    
                    # Draw lines between points
                    for i in range(len(widget_points) - 1):
                        painter.drawLine(
                            widget_points[i][0], widget_points[i][1],
                            widget_points[i+1][0], widget_points[i+1][1]
                        )
                    
                    # Draw a line back to the first point if drawing a polygon
                    if len(widget_points) >= 3 and self.zone_editor.is_drawing():
                        painter.drawLine(
                            widget_points[-1][0], widget_points[-1][1],
                            widget_points[0][0], widget_points[0][1]
                        )
                    
                    # Draw points
                    for wx, wy in widget_points:
                        painter.setBrush(QColor(255, 0, 0))
                        painter.drawEllipse(wx - 5, wy - 5, 10, 10)

    def _get_widget_coordinates(self, image_x, image_y):
        """Convert image coordinates to widget coordinates.
        
        Args:
            image_x (int): X coordinate in image space
            image_y (int): Y coordinate in image space
            
        Returns:
            tuple: (x, y) coordinates in widget space, or (None, None) if outside widget
        """
        if self.current_frame is None:
            return None, None
        
        # Get display label geometry
        label_rect = self.display_label.rect()
        pixmap = self.display_label.pixmap()
        
        if not pixmap:
            return None, None
        
        # Calculate the image position and size within the label
        pixmap_size = pixmap.size()
        
        # Get the position of the image within the label
        x_offset = (label_rect.width() - pixmap_size.width()) / 2
        y_offset = (label_rect.height() - pixmap_size.height()) / 2
        
        # Calculate widget coordinates
        widget_x = int(self.display_label.x() + x_offset + (image_x * pixmap_size.width() / self.current_frame.shape[1]))
        widget_y = int(self.display_label.y() + y_offset + (image_y * pixmap_size.height() / self.current_frame.shape[0]))
        
        return widget_x, widget_y
    
    def resizeEvent(self, event):
        """Handle resize events for the widget.
        
        Args:
            event (QResizeEvent): The resize event
        """
        super().resizeEvent(event)
        # Update the display if we have a current frame
        if self.current_frame is not None:
            self._update_frame()