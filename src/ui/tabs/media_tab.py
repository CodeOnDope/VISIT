"""
Media tab for the VISIT Museum Tracker application.

This module implements the media tab UI that allows users to manage and
view media files associated with the detection system.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem, QFileDialog,
    QTabWidget, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap

from src.utils.logger import setup_logger


class MediaTab(QWidget):
    """Tab for managing and viewing media files."""
    
    def __init__(self, parent=None):
        """Initialize the media tab.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Set up logging
        self.logger = setup_logger("MediaTab", level=logging.INFO)
        
        # Initialize media directories
        self.image_dir = os.path.join("media", "active", "images")
        self.video_dir = os.path.join("media", "active", "videos")
        
        # Ensure directories exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Set up UI
        self._init_ui()
        
        # Load media files
        self._load_media_files()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tabs for different media types
        self.media_tabs = QTabWidget()
        
        # Images tab
        images_widget = self._create_images_tab()
        self.media_tabs.addTab(images_widget, "Images")
        
        # Videos tab
        videos_widget = self._create_videos_tab()
        self.media_tabs.addTab(videos_widget, "Videos")
        
        # Add tabs to main layout
        main_layout.addWidget(self.media_tabs)
    
    def _create_images_tab(self):
        """Create images tab.
        
        Returns:
            QWidget: The images tab widget
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Image list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.import_image_button = QPushButton("Import Image")
        self.import_image_button.clicked.connect(self._import_image)
        controls_layout.addWidget(self.import_image_button)
        
        self.delete_image_button = QPushButton("Delete Image")
        self.delete_image_button.clicked.connect(self._delete_image)
        controls_layout.addWidget(self.delete_image_button)
        
        # Add category filter
        controls_layout.addWidget(QLabel("Category:"))
        self.image_category_combo = QComboBox()
        self.image_category_combo.addItems(["All", "Faces", "Expressions", "Groups", "Other"])
        controls_layout.addWidget(self.image_category_combo)
        
        left_layout.addLayout(controls_layout)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(80, 60))
        self.image_list.currentItemChanged.connect(self._image_selected)
        left_layout.addWidget(self.image_list)
        
        # Right panel - Image preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Preview label
        self.image_preview_label = QLabel("No image selected")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.image_preview_label)
        
        # Image info
        self.image_info_label = QLabel("")
        right_layout.addWidget(self.image_info_label)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes (40% list, 60% preview)
        splitter.setSizes([400, 600])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        return widget
    
    def _create_videos_tab(self):
        """Create videos tab.
        
        Returns:
            QWidget: The videos tab widget
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Video list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.import_video_button = QPushButton("Import Video")
        self.import_video_button.clicked.connect(self._import_video)
        controls_layout.addWidget(self.import_video_button)
        
        self.delete_video_button = QPushButton("Delete Video")
        self.delete_video_button.clicked.connect(self._delete_video)
        controls_layout.addWidget(self.delete_video_button)
        
        # Add category filter
        controls_layout.addWidget(QLabel("Category:"))
        self.video_category_combo = QComboBox()
        self.video_category_combo.addItems(["All", "Sessions", "Calibration", "Demos", "Other"])
        controls_layout.addWidget(self.video_category_combo)
        
        left_layout.addLayout(controls_layout)
        
        # Video list
        self.video_list = QListWidget()
        self.video_list.setIconSize(QSize(80, 60))
        self.video_list.currentItemChanged.connect(self._video_selected)
        left_layout.addWidget(self.video_list)
        
        # Right panel - Video player (placeholder)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Video controls (placeholder)
        video_controls = QHBoxLayout()
        
        play_button = QPushButton("Play")
        video_controls.addWidget(play_button)
        
        pause_button = QPushButton("Pause")
        video_controls.addWidget(pause_button)
        
        stop_button = QPushButton("Stop")
        video_controls.addWidget(stop_button)
        
        # Video preview (placeholder)
        self.video_preview_label = QLabel("No video selected")
        self.video_preview_label.setAlignment(Qt.AlignCenter)
        self.video_preview_label.setStyleSheet("background-color: black;")
        right_layout.addWidget(self.video_preview_label)
        
        # Add video controls
        right_layout.addLayout(video_controls)
        
        # Video info
        self.video_info_label = QLabel("")
        right_layout.addWidget(self.video_info_label)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes (40% list, 60% preview)
        splitter.setSizes([400, 600])
        
        # Add splitter to layout
        layout.addWidget(splitter)
        
        return widget
    
    def _load_media_files(self):
        """Load media files from disk."""
        # Clear existing items
        self.image_list.clear()
        self.video_list.clear()
        
        # Load images
        if os.path.exists(self.image_dir):
            for filename in os.listdir(self.image_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self._add_image_to_list(filename)
        
        # Load videos
        if os.path.exists(self.video_dir):
            for filename in os.listdir(self.video_dir):
                if filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    self._add_video_to_list(filename)
    
    def _add_image_to_list(self, filename):
        """Add an image to the image list.
        
        Args:
            filename (str): Image filename
        """
        # Create a list item with icon and text
        item = QListWidgetItem(filename)
        
        # Add to list
        self.image_list.addItem(item)
    
    def _add_video_to_list(self, filename):
        """Add a video to the video list.
        
        Args:
            filename (str): Video filename
        """
        # Create a list item with icon and text
        item = QListWidgetItem(filename)
        
        # Add to list
        self.video_list.addItem(item)
    
    def _import_image(self):
        """Import an image file."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if not file_paths:
            return
        
        for file_path in file_paths:
            # Get filename
            filename = os.path.basename(file_path)
            
            # Destination path
            dest_path = os.path.join(self.image_dir, filename)
            
            # Check if file already exists
            if os.path.exists(dest_path):
                response = QMessageBox.question(
                    self, "File Exists",
                    f"File {filename} already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if response != QMessageBox.Yes:
                    continue
            
            try:
                # Copy file (simplified, should use shutil in real implementation)
                with open(file_path, 'rb') as src_file:
                    with open(dest_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                
                # Add to list
                self._add_image_to_list(filename)
                
                self.logger.info(f"Imported image: {filename}")
            
            except Exception as e:
                self.logger.error(f"Error importing image {filename}: {e}")
                QMessageBox.warning(self, "Import Error", f"Error importing {filename}: {str(e)}")
    
    def _import_video(self):
        """Import a video file."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        
        if not file_paths:
            return
        
        for file_path in file_paths:
            # Get filename
            filename = os.path.basename(file_path)
            
            # Destination path
            dest_path = os.path.join(self.video_dir, filename)
            
            # Check if file already exists
            if os.path.exists(dest_path):
                response = QMessageBox.question(
                    self, "File Exists",
                    f"File {filename} already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if response != QMessageBox.Yes:
                    continue
            
            try:
                # Copy file (simplified, should use shutil in real implementation)
                with open(file_path, 'rb') as src_file:
                    with open(dest_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                
                # Add to list
                self._add_video_to_list(filename)
                
                self.logger.info(f"Imported video: {filename}")
            
            except Exception as e:
                self.logger.error(f"Error importing video {filename}: {e}")
                QMessageBox.warning(self, "Import Error", f"Error importing {filename}: {str(e)}")
    
    def _delete_image(self):
        """Delete the selected image."""
        current_item = self.image_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Delete Image", "No image selected")
            return
        
        filename = current_item.text()
        file_path = os.path.join(self.image_dir, filename)
        
        # Confirm deletion
        response = QMessageBox.question(
            self, "Delete Image",
            f"Are you sure you want to delete {filename}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response != QMessageBox.Yes:
            return
        
        try:
            # Delete file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from list
            self.image_list.takeItem(self.image_list.row(current_item))
            
            # Clear preview if it was showing the deleted image
            self.image_preview_label.setText("No image selected")
            self.image_info_label.setText("")
            
            self.logger.info(f"Deleted image: {filename}")
        
        except Exception as e:
            self.logger.error(f"Error deleting image {filename}: {e}")
            QMessageBox.warning(self, "Delete Error", f"Error deleting {filename}: {str(e)}")
    
    def _delete_video(self):
        """Delete the selected video."""
        current_item = self.video_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Delete Video", "No video selected")
            return
        
        filename = current_item.text()
        file_path = os.path.join(self.video_dir, filename)
        
        # Confirm deletion
        response = QMessageBox.question(
            self, "Delete Video",
            f"Are you sure you want to delete {filename}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response != QMessageBox.Yes:
            return
        
        try:
            # Delete file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Remove from list
            self.video_list.takeItem(self.video_list.row(current_item))
            
            # Clear preview if it was showing the deleted video
            self.video_preview_label.setText("No video selected")
            self.video_info_label.setText("")
            
            self.logger.info(f"Deleted video: {filename}")
        
        except Exception as e:
            self.logger.error(f"Error deleting video {filename}: {e}")
            QMessageBox.warning(self, "Delete Error", f"Error deleting {filename}: {str(e)}")
    
    def _image_selected(self, current, previous):
        """Handle image selection event.
        
        Args:
            current (QListWidgetItem): Current selected item
            previous (QListWidgetItem): Previous selected item
        """
        if not current:
            # No selection
            self.image_preview_label.setText("No image selected")
            self.image_info_label.setText("")
            return
        
        filename = current.text()
        file_path = os.path.join(self.image_dir, filename)
        
        # Update preview (placeholder implementation)
        self.image_preview_label.setText(f"Preview of: {filename}")
        
        # Update info
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            self.image_info_label.setText(f"Filename: {filename}\nSize: {size_kb:.1f} KB")
        else:
            self.image_info_label.setText(f"Filename: {filename}\nFile not found")
    
    def _video_selected(self, current, previous):
        """Handle video selection event.
        
        Args:
            current (QListWidgetItem): Current selected item
            previous (QListWidgetItem): Previous selected item
        """
        if not current:
            # No selection
            self.video_preview_label.setText("No video selected")
            self.video_info_label.setText("")
            return
        
        filename = current.text()
        file_path = os.path.join(self.video_dir, filename)
        
        # Update preview (placeholder implementation)
        self.video_preview_label.setText(f"Preview of: {filename}")
        
        # Update info
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.video_info_label.setText(f"Filename: {filename}\nSize: {size_mb:.1f} MB")
        else:
            self.video_info_label.setText(f"Filename: {filename}\nFile not found")
    
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
        # Reload media files when tab is selected
        self._load_media_files()
