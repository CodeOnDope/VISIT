#!/usr/bin/env python3
"""
VISIT-Museum-Tracker System
A comprehensive visitor analysis system for museums using MediaPipe, OpenCV, and PyQt5
"""

import sys
import cv2
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QCheckBox,
    QGroupBox, QGridLayout, QTextEdit, QProgressBar, QComboBox,
    QTableWidget, QTableWidgetItem, QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Some features will be disabled.")

@dataclass
class VisitorData:
    """Data structure for visitor information"""
    visitor_id: str
    timestamp: datetime
    position: Tuple[int, int]
    emotion: str
    engagement_level: float
    zone: str
    duration: float

class DatabaseManager:
    """Handles all database operations for visitor analytics"""
    
    def __init__(self, db_path: str = "museum_analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create visitors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT,
                timestamp DATETIME,
                x_position INTEGER,
                y_position INTEGER,
                emotion TEXT,
                engagement_level REAL,
                zone TEXT,
                duration REAL
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                start_time DATETIME,
                end_time DATETIME,
                total_visitors INTEGER,
                avg_engagement REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_visitor_data(self, data: VisitorData):
        """Insert visitor data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO visitors (visitor_id, timestamp, x_position, y_position, 
                                emotion, engagement_level, zone, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data.visitor_id, data.timestamp, data.position[0], data.position[1],
              data.emotion, data.engagement_level, data.zone, data.duration))
        
        conn.commit()
        conn.close()
    
    def get_analytics_data(self, hours: int = 24) -> Dict:
        """Retrieve analytics data for the specified time period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT COUNT(*) as total_visitors,
                   AVG(engagement_level) as avg_engagement,
                   COUNT(DISTINCT zone) as zones_visited
            FROM visitors 
            WHERE timestamp > ?
        ''', (since,))
        
        result = cursor.fetchone()
        
        analytics = {
            'total_visitors': result[0] if result[0] else 0,
            'avg_engagement': result[1] if result[1] else 0.0,
            'zones_visited': result[2] if result[2] else 0
        }
        
        conn.close()
        return analytics

class DetectorEngine:
    """Core detection engine using MediaPipe"""
    
    def __init__(self):
        self.mp_face_detection = None
        self.mp_pose = None
        self.mp_hands = None
        self.mp_drawing = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        if not MEDIAPIPE_AVAILABLE or not self.face_detection:
            return []
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        
        return faces
    
    def detect_pose(self, image):
        """Detect pose landmarks in the image"""
        if not MEDIAPIPE_AVAILABLE or not self.pose:
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        return results.pose_landmarks
    
    def detect_hands(self, image):
        """Detect hand landmarks in the image"""
        if not MEDIAPIPE_AVAILABLE or not self.hands:
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        return results.multi_hand_landmarks
    
    def analyze_engagement(self, faces, poses, hands) -> float:
        """Analyze visitor engagement based on detected features"""
        engagement_score = 0.5  # Base engagement
        
        # Face detection increases engagement
        if faces:
            engagement_score += 0.2
        
        # Pose detection (active posture)
        if poses:
            engagement_score += 0.2
        
        # Hand gestures (interaction)
        if hands:
            engagement_score += 0.3
        
        return min(engagement_score, 1.0)

class CameraThread(QThread):
    """Thread for camera capture and processing"""
    
    frame_ready = pyqtSignal(np.ndarray)
    detection_data = pyqtSignal(dict)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.detector = DetectorEngine()
        self.running = False
        self.cap = None
    
    def start_capture(self):
        """Start camera capture"""
        self.running = True
        self.start()
    
    def stop_capture(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def run(self):
        """Main camera processing loop"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect features
            faces = self.detector.detect_faces(frame)
            poses = self.detector.detect_pose(frame)
            hands = self.detector.detect_hands(frame)
            
            # Calculate engagement
            engagement = self.detector.analyze_engagement(faces, poses, hands)
            
            # Draw detections on frame
            annotated_frame = frame.copy()
            
            # Draw face bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(annotated_frame, 'Face', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw pose landmarks
            if poses and MEDIAPIPE_AVAILABLE:
                self.detector.mp_drawing.draw_landmarks(
                    annotated_frame, poses, self.detector.mp_pose.POSE_CONNECTIONS)
            
            # Draw hand landmarks
            if hands and MEDIAPIPE_AVAILABLE:
                for hand_landmarks in hands:
                    self.detector.mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks, self.detector.mp_hands.HAND_CONNECTIONS)
            
            # Emit signals
            self.frame_ready.emit(annotated_frame)
            self.detection_data.emit({
                'faces': len(faces),
                'engagement': engagement,
                'timestamp': datetime.now()
            })
            
            time.sleep(0.03)  # ~30 FPS

class CameraWidget(QWidget):
    """Widget for displaying camera feed"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.detection_data.connect(self.update_detection_data)
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Camera Feed")
        
        # Controls
        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        # Status
        self.status_label = QLabel("Status: Ready")
        
        layout.addWidget(self.video_label)
        layout.addLayout(controls_layout)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update the video frame display"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    @pyqtSlot(dict)
    def update_detection_data(self, data):
        """Update detection status"""
        faces = data.get('faces', 0)
        engagement = data.get('engagement', 0.0)
        self.status_label.setText(f"Faces: {faces} | Engagement: {engagement:.2f}")
    
    def start_camera(self):
        """Start camera capture"""
        self.camera_thread.start_capture()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Recording")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_thread.stop_capture()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")

class AnalyticsWidget(QWidget):
    """Widget for displaying analytics and statistics"""
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_analytics)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Visitor Analytics Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        # Stats grid
        stats_group = QGroupBox("Current Statistics")
        stats_layout = QGridLayout()
        
        self.total_visitors_label = QLabel("Total Visitors: 0")
        self.avg_engagement_label = QLabel("Avg Engagement: 0.00")
        self.zones_visited_label = QLabel("Zones Visited: 0")
        self.current_time_label = QLabel(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        stats_layout.addWidget(self.total_visitors_label, 0, 0)
        stats_layout.addWidget(self.avg_engagement_label, 0, 1)
        stats_layout.addWidget(self.zones_visited_label, 1, 0)
        stats_layout.addWidget(self.current_time_label, 1, 1)
        
        stats_group.setLayout(stats_layout)
        
        # Engagement chart (placeholder)
        chart_group = QGroupBox("Engagement Over Time")
        chart_layout = QVBoxLayout()
        self.chart_label = QLabel("Engagement visualization would appear here")
        self.chart_label.setMinimumHeight(200)
        self.chart_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.chart_label.setAlignment(Qt.AlignCenter)
        chart_layout.addWidget(self.chart_label)
        chart_group.setLayout(chart_layout)
        
        # Recent activity
        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout()
        self.activity_text = QTextEdit()
        self.activity_text.setMaximumHeight(150)
        self.activity_text.setReadOnly(True)
        activity_layout.addWidget(self.activity_text)
        activity_group.setLayout(activity_layout)
        
        layout.addWidget(title)
        layout.addWidget(stats_group)
        layout.addWidget(chart_group)
        layout.addWidget(activity_group)
        
        self.setLayout(layout)
    
    def update_analytics(self):
        """Update analytics display"""
        try:
            data = self.db_manager.get_analytics_data()
            
            self.total_visitors_label.setText(f"Total Visitors: {data['total_visitors']}")
            self.avg_engagement_label.setText(f"Avg Engagement: {data['avg_engagement']:.2f}")
            self.zones_visited_label.setText(f"Zones Visited: {data['zones_visited']}")
            self.current_time_label.setText(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
            
            # Add recent activity
            activity_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Analytics updated - " \
                          f"Visitors: {data['total_visitors']}, Engagement: {data['avg_engagement']:.2f}\n"
            self.activity_text.append(activity_msg)
            
        except Exception as e:
            print(f"Error updating analytics: {e}")

class ConfigWidget(QWidget):
    """Configuration widget for system settings"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Camera settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QGridLayout()
        
        camera_layout.addWidget(QLabel("Camera ID:"), 0, 0)
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        camera_layout.addWidget(self.camera_id_spin, 0, 1)
        
        camera_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        camera_layout.addWidget(self.resolution_combo, 1, 1)
        
        camera_group.setLayout(camera_layout)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout()
        
        detection_layout.addWidget(QLabel("Face Detection Confidence:"), 0, 0)
        self.face_confidence_slider = QSlider(Qt.Horizontal)
        self.face_confidence_slider.setRange(10, 100)
        self.face_confidence_slider.setValue(50)
        detection_layout.addWidget(self.face_confidence_slider, 0, 1)
        
        detection_layout.addWidget(QLabel("Enable Pose Detection:"), 1, 0)
        self.pose_checkbox = QCheckBox()
        self.pose_checkbox.setChecked(True)
        detection_layout.addWidget(self.pose_checkbox, 1, 1)
        
        detection_layout.addWidget(QLabel("Enable Hand Detection:"), 2, 0)
        self.hand_checkbox = QCheckBox()
        self.hand_checkbox.setChecked(True)
        detection_layout.addWidget(self.hand_checkbox, 2, 1)
        
        detection_group.setLayout(detection_layout)
        
        # Zone settings
        zone_group = QGroupBox("Zone Configuration")
        zone_layout = QVBoxLayout()
        zone_layout.addWidget(QLabel("Zone configuration would be implemented here"))
        zone_group.setLayout(zone_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        reset_btn = QPushButton("Reset to Defaults")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        
        layout.addWidget(camera_group)
        layout.addWidget(detection_group)
        layout.addWidget(zone_group)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("VISIT-Museum-Tracker v1.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("VISIT-Museum-Tracker System")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; padding: 10px;")
        
        # Tab widget
        tab_widget = QTabWidget()
        
        # Camera tab
        self.camera_widget = CameraWidget()
        tab_widget.addTab(self.camera_widget, "📹 Live Feed")
        
        # Analytics tab
        self.analytics_widget = AnalyticsWidget(self.db_manager)
        tab_widget.addTab(self.analytics_widget, "📊 Analytics")
        
        # Configuration tab
        self.config_widget = ConfigWidget()
        tab_widget.addTab(self.config_widget, "⚙️ Settings")
        
        # About tab
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml("""
        <h2>VISIT-Museum-Tracker System</h2>
        <p><strong>Version:</strong> 1.0</p>
        <p><strong>Purpose:</strong> Real-time visitor analysis for museums</p>
        
        <h3>Features:</h3>
        <ul>
            <li>Real-time face detection and tracking</li>
            <li>Pose and gesture analysis</li>
            <li>Engagement level measurement</li>
            <li>Analytics dashboard</li>
            <li>SQLite database for data storage</li>
        </ul>
        
        <h3>Technology Stack:</h3>
        <ul>
            <li>MediaPipe for computer vision</li>
            <li>OpenCV for image processing</li>
            <li>PyQt5 for user interface</li>
            <li>SQLite for data storage</li>
        </ul>
        
        <p><em>Developed for museum environments to provide insights into visitor behavior and engagement.</em></p>
        """)
        about_layout.addWidget(about_text)
        about_widget.setLayout(about_layout)
        tab_widget.addTab(about_widget, "ℹ️ About")
        
        layout.addWidget(header)
        layout.addWidget(tab_widget)
        
        central_widget.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("System Ready")
        
        # Apply styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #bdc3c7;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin: 10px 0;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Check MediaPipe availability
    if not MEDIAPIPE_AVAILABLE:
        print("Warning: MediaPipe not available. Install with: pip install mediapipe")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()