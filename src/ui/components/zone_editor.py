 
"""
Zone editor component for the VISIT Museum Tracker system.

This module implements a zone editor widget that allows users to create,
edit, and manage zones for the zone detector.
"""

import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QColorDialog, QListWidget, QListWidgetItem,
    QMessageBox, QFormLayout, QGroupBox, QMenu, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QBrush, QCursor


class ZoneEditor(QWidget):
    """Widget for creating and editing zones."""
    
    # Signals
    zone_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the zone editor widget.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Initialize state
        self.zones = {}
        self.current_zone_id = None
        self.editing = False
        self.drawing = False
        self.edit_points = []
        self.frame_size = (640, 480)
        
        # Set up UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Left panel - Zone list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Zone list
        list_group = QGroupBox("Zones")
        list_layout = QVBoxLayout(list_group)
        
        self.zone_list = QListWidget()
        self.zone_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.zone_list.customContextMenuRequested.connect(self._show_context_menu)
        self.zone_list.currentItemChanged.connect(self._on_zone_selected)
        list_layout.addWidget(self.zone_list)
        
        # Zone list buttons
        buttons_layout = QHBoxLayout()
        
        self.add_zone_button = QPushButton("Add Zone")
        self.add_zone_button.clicked.connect(self._start_add_zone)
        buttons_layout.addWidget(self.add_zone_button)
        
        self.edit_zone_button = QPushButton("Edit")
        self.edit_zone_button.clicked.connect(self._start_edit_zone)
        self.edit_zone_button.setEnabled(False)
        buttons_layout.addWidget(self.edit_zone_button)
        
        self.delete_zone_button = QPushButton("Delete")
        self.delete_zone_button.clicked.connect(self._delete_zone)
        self.delete_zone_button.setEnabled(False)
        buttons_layout.addWidget(self.delete_zone_button)
        
        list_layout.addLayout(buttons_layout)
        
        left_layout.addWidget(list_group)
        
        # Zone properties
        properties_group = QGroupBox("Zone Properties")
        properties_layout = QFormLayout(properties_group)
        
        # Zone name
        self.zone_name_edit = QLineEdit()
        self.zone_name_edit.setEnabled(False)
        self.zone_name_edit.textChanged.connect(self._update_zone_name)
        properties_layout.addRow("Name:", self.zone_name_edit)
        
        # Zone color
        color_layout = QHBoxLayout()
        self.zone_color_button = QPushButton()
        self.zone_color_button.setEnabled(False)
        self.zone_color_button.setFixedSize(24, 24)
        self.zone_color_button.setStyleSheet("background-color: #00FF00;")
        self.zone_color_button.clicked.connect(self._select_zone_color)
        color_layout.addWidget(self.zone_color_button)
        color_layout.addStretch()
        
        properties_layout.addRow("Color:", color_layout)
        
        # Zone statistics
        self.zone_stats_label = QLabel("No zone selected")
        properties_layout.addRow("Statistics:", self.zone_stats_label)
        
        left_layout.addWidget(properties_group)
        
        # Drawing instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        
        instructions_text = QLabel(
            "1. Click 'Add Zone' to create a new zone\n"
            "2. Click on the video to place points\n"
            "3. Press Enter or double-click to complete\n"
            "4. Press Escape to cancel\n\n"
            "To edit, select a zone and click 'Edit'"
        )
        instructions_layout.addWidget(instructions_text)
        
        left_layout.addWidget(instructions_group)
        
        # Add left panel to main layout
        main_layout.addWidget(left_panel)
        
        # Set initial state
        self._update_ui_state()
    
    def _update_ui_state(self):
        """Update UI elements based on current state."""
        # Enable/disable controls based on selection
        has_selection = self.current_zone_id is not None
        self.zone_name_edit.setEnabled(has_selection)
        self.zone_color_button.setEnabled(has_selection)
        self.edit_zone_button.setEnabled(has_selection and not self.editing)
        self.delete_zone_button.setEnabled(has_selection and not self.editing)
        
        # Enable/disable controls based on editing state
        self.add_zone_button.setEnabled(not self.editing)
        self.zone_list.setEnabled(not self.editing)
        
        # Update zone properties
        if has_selection and self.current_zone_id in self.zones:
            zone = self.zones[self.current_zone_id]
            self.zone_name_edit.setText(zone["name"])
            
            # Set color button background
            color = zone["color"]
            self.zone_color_button.setStyleSheet(
                f"background-color: rgb({color[2]}, {color[1]}, {color[0]});"
            )
            
            # Update statistics
            stats_text = f"Visits: {zone.get('visit_count', 0)}"
            self.zone_stats_label.setText(stats_text)
        else:
            self.zone_name_edit.setText("")
            self.zone_color_button.setStyleSheet("background-color: #00FF00;")
            self.zone_stats_label.setText("No zone selected")
    
    def _show_context_menu(self, position):
        """Show context menu for zone list.
        
        Args:
            position (QPoint): Position where the menu should appear
        """
        if self.editing or not self.zone_list.count():
            return
        
        menu = QMenu()
        edit_action = QAction("Edit", self)
        edit_action.triggered.connect(self._start_edit_zone)
        menu.addAction(edit_action)
        
        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.zone_name_edit.setFocus())
        menu.addAction(rename_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self._delete_zone)
        menu.addAction(delete_action)
        
        menu.exec_(self.zone_list.mapToGlobal(position))
    
    def _on_zone_selected(self, current, previous):
        """Handle zone selection change.
        
        Args:
            current (QListWidgetItem): Current selected item
            previous (QListWidgetItem): Previous selected item
        """
        if current:
            self.current_zone_id = current.data(Qt.UserRole)
        else:
            self.current_zone_id = None
        
        self._update_ui_state()
    
    def _update_zone_name(self, name):
        """Update the name of the current zone.
        
        Args:
            name (str): New zone name
        """
        if not self.current_zone_id or self.current_zone_id not in self.zones:
            return
        
        # Update zone name
        self.zones[self.current_zone_id]["name"] = name
        
        # Update list item
        for i in range(self.zone_list.count()):
            item = self.zone_list.item(i)
            if item.data(Qt.UserRole) == self.current_zone_id:
                item.setText(name)
                break
        
        # Emit signal
        self.zone_changed.emit()
    
    def _select_zone_color(self):
        """Open color dialog to select zone color."""
        if not self.current_zone_id or self.current_zone_id not in self.zones:
            return
        
        # Get current color
        current_color = self.zones[self.current_zone_id]["color"]
        initial_color = QColor(current_color[2], current_color[1], current_color[0])
        
        # Open color dialog
        color = QColorDialog.getColor(initial_color, self, "Select Zone Color")
        
        if color.isValid():
            # Update zone color (BGR format for OpenCV)
            new_color = (color.blue(), color.green(), color.red())
            self.zones[self.current_zone_id]["color"] = new_color
            
            # Update color button
            self.zone_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            )
            
            # Update list item background
            for i in range(self.zone_list.count()):
                item = self.zone_list.item(i)
                if item.data(Qt.UserRole) == self.current_zone_id:
                    item.setBackground(QBrush(color))
                    break
            
            # Emit signal
            self.zone_changed.emit()
    
    def _start_add_zone(self):
        """Start adding a new zone."""
        self.editing = True
        self.drawing = True
        self.edit_points = []
        
        # Create a temporary zone ID
        self.current_zone_id = f"zone_{len(self.zones)}"
        
        # Update UI state
        self._update_ui_state()
        
        # Emit signal to notify parent that we're now in drawing mode
        self.zone_changed.emit()
    
    def _start_edit_zone(self):
        """Start editing the selected zone."""
        if not self.current_zone_id or self.current_zone_id not in self.zones:
            return
        
        self.editing = True
        self.drawing = False
        
        # Get current points
        points = self.zones[self.current_zone_id]["points"]
        self.edit_points = points.reshape(-1, 2).tolist()
        
        # Update UI state
        self._update_ui_state()
        
        # Emit signal
        self.zone_changed.emit()
    
    def _delete_zone(self):
        """Delete the selected zone."""
        if not self.current_zone_id or self.current_zone_id not in self.zones:
            return
        
        # Confirm deletion
        result = QMessageBox.question(
            self, "Delete Zone",
            f"Are you sure you want to delete the zone '{self.zones[self.current_zone_id]['name']}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Remove from zones dictionary
            del self.zones[self.current_zone_id]
            
            # Remove from list
            for i in range(self.zone_list.count()):
                item = self.zone_list.item(i)
                if item.data(Qt.UserRole) == self.current_zone_id:
                    self.zone_list.takeItem(i)
                    break
            
            # Clear selection
            self.current_zone_id = None
            
            # Update UI state
            self._update_ui_state()
            
            # Emit signal
            self.zone_changed.emit()
    
    def handle_mouse_event(self, event_type, x, y):
        """Handle mouse events from the video display.
        
        Args:
            event_type (str): Type of event ('press', 'move', 'release', 'double_click')
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if the event was handled, False otherwise
        """
        if not self.editing:
            return False
        
        if self.drawing:
            # Adding a new zone
            if event_type == 'press':
                # Add point
                self.edit_points.append([x, y])
                return True
            
            elif event_type == 'double_click' or event_type == 'enter_key':
                # Complete polygon if we have at least 3 points
                if len(self.edit_points) >= 3:
                    # Create zone
                    zone_name = f"Zone {len(self.zones) + 1}"
                    zone_color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    
                    # Add zone
                    self.add_zone(
                        self.current_zone_id,
                        zone_name,
                        self.edit_points,
                        zone_color
                    )
                    
                    # Reset state
                    self.editing = False
                    self.drawing = False
                    self.edit_points = []
                    
                    # Update UI state
                    self._update_ui_state()
                    
                    # Emit signal
                    self.zone_changed.emit()
                    
                    return True
            
            elif event_type == 'escape_key':
                # Cancel drawing
                self.editing = False
                self.drawing = False
                self.edit_points = []
                self.current_zone_id = None
                
                # Update UI state
                self._update_ui_state()
                
                # Emit signal
                self.zone_changed.emit()
                
                return True
        
        else:
            # Editing an existing zone
            if event_type == 'press':
                # Check if we're near an existing point
                for i, point in enumerate(self.edit_points):
                    if (x - point[0])**2 + (y - point[1])**2 < 100:  # Within 10 pixels
                        # Move this point
                        self.edit_points[i] = [x, y]
                        
                        # Update zone
                        self._update_zone_points()
                        
                        return True
            
            elif event_type == 'enter_key':
                # Complete editing
                self.editing = False
                
                # Update UI state
                self._update_ui_state()
                
                # Emit signal
                self.zone_changed.emit()
                
                return True
            
            elif event_type == 'escape_key':
                # Cancel editing
                self.editing = False
                
                # Update UI state
                self._update_ui_state()
                
                # Emit signal
                self.zone_changed.emit()
                
                return True
        
        return False
    
    def _update_zone_points(self):
        """Update the points of the current zone during editing."""
        if not self.current_zone_id or self.current_zone_id not in self.zones:
            return
        
        # Update zone points
        points_array = np.array(self.edit_points, np.int32)
        self.zones[self.current_zone_id]["points"] = points_array.reshape((-1, 1, 2))
        
        # Emit signal
        self.zone_changed.emit()
    
    def add_zone(self, zone_id, name, points, color=(0, 255, 0)):
        """Add a new zone.
        
        Args:
            zone_id (str): Zone identifier
            name (str): Zone name
            points (list): List of (x, y) points defining the zone polygon
            color (tuple, optional): Color for visualization (BGR). Defaults to green.
            
        Returns:
            bool: True if zone was added, False if it already exists
        """
        if zone_id in self.zones:
            return False
        
        # Convert points to numpy array
        points_array = np.array(points, np.int32)
        
        # Create zone
        zone = {
            "id": zone_id,
            "name": name,
            "points": points_array.reshape((-1, 1, 2)),
            "color": color,
            "active": True,
            "visit_count": 0,
            "total_time": 0
        }
        
        # Add zone
        self.zones[zone_id] = zone
        
        # Add to list
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, zone_id)
        
        # Set background color
        item_color = QColor(color[2], color[1], color[0])
        item.setBackground(QBrush(item_color))
        
        self.zone_list.addItem(item)
        self.zone_list.setCurrentItem(item)
        
        return True
    
    def import_zones(self, zones):
        """Import zones from zone detector.
        
        Args:
            zones (dict): Dictionary of zones from zone detector
        """
        # Clear current zones
        self.zones = {}
        self.zone_list.clear()
        
        # Import zones
        for zone_id, zone in zones.items():
            self.add_zone(
                zone_id,
                zone["name"],
                zone["points"].reshape(-1, 2).tolist(),
                zone["color"]
            )
    
    def get_zones(self):
        """Get all defined zones.
        
        Returns:
            dict: Dictionary of zones
        """
        return self.zones
    
    def set_frame_size(self, width, height):
        """Set the frame size for coordinates scaling.
        
        Args:
            width (int): Frame width
            height (int): Frame height
        """
        self.frame_size = (width, height)
    
    def get_current_points(self):
        """Get the current edit points.
        
        Returns:
            list: List of (x, y) points
        """
        return self.edit_points
    
    def is_editing(self):
        """Check if the editor is in editing mode.
        
        Returns:
            bool: True if editing, False otherwise
        """
        return self.editing
    
    def is_drawing(self):
        """Check if the editor is in drawing mode.
        
        Returns:
            bool: True if drawing, False otherwise
        """
        return self.drawing