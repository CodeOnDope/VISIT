# File: src/ui/analytics_tab.py

# Add to src/ui/analytics_tab.py

import os
import logging
import json
import csv
import cv2
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QTabWidget, QPushButton, QFileDialog, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

from src.utils.analytics_db import AnalyticsDB

logger = logging.getLogger("AnalyticsTab")

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class AnalyticsTab(QWidget):
    def __init__(self, parent=None):
        super(AnalyticsTab, self).__init__(parent)
        self.parent = parent
        self.db = AnalyticsDB()
        
        self.init_ui()
        
        # Set up a timer to periodically refresh data
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
        
        # Initial data load
        self.refresh_data()
        
    def init_ui(self):
        """Initialize the analytics dashboard UI."""
        layout = QVBoxLayout()
        
        # Time period selection
        time_period_layout = QHBoxLayout()
        time_period_layout.addWidget(QLabel("Time Period:"))
        self.time_period_combo = QComboBox()
        self.time_period_combo.addItems(["Hour", "Day", "Week", "Month"])
        self.time_period_combo.setCurrentIndex(1)  # Default to "Day"
        self.time_period_combo.currentIndexChanged.connect(self.refresh_data)
        time_period_layout.addWidget(self.time_period_combo)
        
        # Export button
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        time_period_layout.addWidget(self.export_btn)
        
        time_period_layout.addStretch()
        layout.addLayout(time_period_layout)
        
        # Create tabs for different charts
        self.tabs = QTabWidget()
        
        # Visitor count tab
        self.visitor_count_tab = QWidget()
        visitor_count_layout = QVBoxLayout()
        self.visitor_count_canvas = MplCanvas(width=8, height=4, dpi=100)
        visitor_count_layout.addWidget(self.visitor_count_canvas)
        self.visitor_count_tab.setLayout(visitor_count_layout)
        self.tabs.addTab(self.visitor_count_tab, "Visitor Count")
        
        # Zone heatmap tab
        self.zone_heatmap_tab = QWidget()
        zone_heatmap_layout = QVBoxLayout()
        self.zone_selection_combo = QComboBox()
        self.zone_selection_combo.currentIndexChanged.connect(self.update_zone_charts)
        zone_heatmap_layout.addWidget(QLabel("Select Zone:"))
        zone_heatmap_layout.addWidget(self.zone_selection_combo)
        
        zone_charts_layout = QHBoxLayout()
        
        # Zone activity chart
        self.zone_activity_group = QGroupBox("Visitor Count by Zone")
        zone_activity_layout = QVBoxLayout()
        self.zone_activity_canvas = MplCanvas(width=4, height=3, dpi=100)
        zone_activity_layout.addWidget(self.zone_activity_canvas)
        self.zone_activity_group.setLayout(zone_activity_layout)
        zone_charts_layout.addWidget(self.zone_activity_group)
        
        # Dwell time chart
        self.dwell_time_group = QGroupBox("Average Dwell Time by Zone")
        dwell_time_layout = QVBoxLayout()
        self.dwell_time_canvas = MplCanvas(width=4, height=3, dpi=100)
        dwell_time_layout.addWidget(self.dwell_time_canvas)
        self.dwell_time_group.setLayout(dwell_time_layout)
        zone_charts_layout.addWidget(self.dwell_time_group)
        
        zone_heatmap_layout.addLayout(zone_charts_layout)
        self.zone_heatmap_tab.setLayout(zone_heatmap_layout)
        self.tabs.addTab(self.zone_heatmap_tab, "Zone Analytics")
        
        # Expression analysis tab
        self.expression_tab = QWidget()
        expression_layout = QVBoxLayout()
        self.expression_canvas = MplCanvas(width=8, height=4, dpi=100)
        expression_layout.addWidget(self.expression_canvas)
        self.expression_tab.setLayout(expression_layout)
        self.tabs.addTab(self.expression_tab, "Expression Analysis")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)

def initialize_db(self):
    """Create database tables if they don't exist."""
    try:
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create visitor_count table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitor_count (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            count INTEGER NOT NULL,
            camera_id TEXT DEFAULT 'default'
        )
        ''')
        
        # More tables...
        
        # Add the visitor_trajectories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS visitor_trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            visitor_id TEXT NOT NULL,
            trajectory TEXT NOT NULL,
            camera_id TEXT DEFAULT 'default'
        )
        ''')
        
        self.conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")


    def refresh_data(self):
        """Refresh all the analytics charts with current data."""
        logger.info("Refreshing analytics data")
        
        # Get the selected time period
        time_period = self.time_period_combo.currentText().lower()
        
        # Update visitor count chart
        self.update_visitor_count_chart(time_period)
        
        # Update zone data and charts
        self.update_zone_selector()
        self.update_zone_charts()
        
        # Update expression analysis chart
        self.update_expression_chart(time_period)
    
    def update_visitor_count_chart(self, time_period):
        """Update the visitor count chart."""
        data = self.db.get_visitor_counts(time_period)
        
        if not data:
            logger.warning("No visitor count data available")
            return
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'count'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Clear the current plot
        self.visitor_count_canvas.axes.clear()
        
        # Plot the visitor count
        self.visitor_count_canvas.axes.plot(df['timestamp'], df['count'], 'b-')
        self.visitor_count_canvas.axes.set_title('Visitor Count Over Time')
        self.visitor_count_canvas.axes.set_xlabel('Time')
        self.visitor_count_canvas.axes.set_ylabel('Number of Visitors')
        self.visitor_count_canvas.axes.grid(True)
        
        # Format x-axis based on the time period
        if time_period == 'hour':
            self.visitor_count_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif time_period == 'day':
            self.visitor_count_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif time_period == 'week':
            self.visitor_count_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%a %H:%M'))
        elif time_period == 'month':
            self.visitor_count_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        
        self.visitor_count_canvas.fig.tight_layout()
        self.visitor_count_canvas.draw()
    
    def update_zone_selector(self):
        """Update the zone selection dropdown."""
        # Get unique zone IDs from the database
        data = self.db.get_zone_activity()
        
        if not data:
            logger.warning("No zone activity data available")
            return
        
        # Extract unique zone IDs
        zones = set()
        for _, zone_id, _, _ in data:
            zones.add(zone_id)
        
        # Update the combo box
        current_text = self.zone_selection_combo.currentText()
        self.zone_selection_combo.clear()
        self.zone_selection_combo.addItems(sorted(zones))
        
        # Try to restore the previous selection
        index = self.zone_selection_combo.findText(current_text)
        if index >= 0:
            self.zone_selection_combo.setCurrentIndex(index)
    
    def update_zone_charts(self):
        """Update the zone activity and dwell time charts."""
        selected_zone = self.zone_selection_combo.currentText()
        if not selected_zone:
            return
        
        time_period = self.time_period_combo.currentText().lower()
        data = self.db.get_zone_activity(time_period)
        
        if not data:
            logger.warning("No zone activity data available")
            return
        
        # Filter data for the selected zone
        df = pd.DataFrame(data, columns=['timestamp', 'zone_id', 'visitor_count', 'dwell_time_ms'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        zone_data = df[df['zone_id'] == selected_zone]
        
        if zone_data.empty:
            logger.warning(f"No data for zone {selected_zone}")
            return
        
        # Update zone activity chart
        self.zone_activity_canvas.axes.clear()
        self.zone_activity_canvas.axes.plot(zone_data['timestamp'], zone_data['visitor_count'], 'g-')
        self.zone_activity_canvas.axes.set_title(f'Visitor Count for Zone: {selected_zone}')
        self.zone_activity_canvas.axes.set_xlabel('Time')
        self.zone_activity_canvas.axes.set_ylabel('Number of Visitors')
        self.zone_activity_canvas.axes.grid(True)
        
        # Format x-axis based on the time period
        if time_period == 'hour':
            self.zone_activity_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif time_period == 'day':
            self.zone_activity_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif time_period == 'week':
            self.zone_activity_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%a %H:%M'))
        elif time_period == 'month':
            self.zone_activity_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        
        self.zone_activity_canvas.fig.tight_layout()
        self.zone_activity_canvas.draw()
        
        # Update dwell time chart
        self.dwell_time_canvas.axes.clear()
        # Convert milliseconds to seconds for better readability
        zone_data['dwell_time_sec'] = zone_data['dwell_time_ms'] / 1000
        self.dwell_time_canvas.axes.plot(zone_data['timestamp'], zone_data['dwell_time_sec'], 'r-')
        self.dwell_time_canvas.axes.set_title(f'Dwell Time for Zone: {selected_zone}')
        self.dwell_time_canvas.axes.set_xlabel('Time')
        self.dwell_time_canvas.axes.set_ylabel('Dwell Time (seconds)')
        self.dwell_time_canvas.axes.grid(True)
        
        # Format x-axis based on the time period
        if time_period == 'hour':
            self.dwell_time_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif time_period == 'day':
            self.dwell_time_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        elif time_period == 'week':
            self.dwell_time_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%a %H:%M'))
        elif time_period == 'month':
            self.dwell_time_canvas.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        
        self.dwell_time_canvas.fig.tight_layout()
        self.dwell_time_canvas.draw()
    
    def update_expression_chart(self, time_period):
        """Update the expression analysis chart."""
        data = self.db.get_expression_data(time_period)
        
        if not data:
            logger.warning("No expression data available")
            return
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'expression', 'count'])
        
        # Group by expression and sum the counts
        expression_totals = df.groupby('expression')['count'].sum()
        
        # Clear the current plot
        self.expression_canvas.axes.clear()
        
        # Create a pie chart
        self.expression_canvas.axes.pie(
            expression_totals, 
            labels=expression_totals.index, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        self.expression_canvas.axes.set_title('Expression Distribution')
        self.expression_canvas.axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        self.expression_canvas.fig.tight_layout()
        self.expression_canvas.draw()
    
    def export_data(self):
        """Export the analytics data to a CSV file."""
        time_period = self.time_period_combo.currentText().lower()
        
        # Ask for the export directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", "", QFileDialog.ShowDirsOnly
        )
        
        if not export_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export visitor count data
        visitor_data = self.db.get_visitor_counts(time_period)
        if visitor_data:
            df = pd.DataFrame(visitor_data, columns=['timestamp', 'count'])
            df.to_csv(os.path.join(export_dir, f"visitor_count_{timestamp}.csv"), index=False)
        
        # Export zone activity data
        zone_data = self.db.get_zone_activity(time_period)
        if zone_data:
            df = pd.DataFrame(zone_data, columns=['timestamp', 'zone_id', 'visitor_count', 'dwell_time_ms'])
            df.to_csv(os.path.join(export_dir, f"zone_activity_{timestamp}.csv"), index=False)
        
        # Export expression data
        expression_data = self.db.get_expression_data(time_period)
        if expression_data:
            df = pd.DataFrame(expression_data, columns=['timestamp', 'expression', 'count'])
            df.to_csv(os.path.join(export_dir, f"expression_data_{timestamp}.csv"), index=False)
        
        logger.info(f"Analytics data exported to {export_dir}")

        # Add this to your analytics_tab.py

def add_heatmap_tab(self):
    """Add a heatmap tab to the analytics dashboard."""
    self.heatmap_tab = QWidget()
    heatmap_layout = QVBoxLayout()
    
    # Controls
    controls_layout = QHBoxLayout()
    controls_layout.addWidget(QLabel("Time Period:"))
    
    self.heatmap_period_combo = QComboBox()
    self.heatmap_period_combo.addItems(["Hour", "Day", "Week", "Month"])
    self.heatmap_period_combo.setCurrentIndex(1)  # Default to "Day"
    self.heatmap_period_combo.currentIndexChanged.connect(self.update_heatmap)
    controls_layout.addWidget(self.heatmap_period_combo)
    
    controls_layout.addWidget(QLabel("Data Type:"))
    self.heatmap_type_combo = QComboBox()
    self.heatmap_type_combo.addItems(["Visitor Count", "Dwell Time", "Expression"])
    self.heatmap_type_combo.currentIndexChanged.connect(self.update_heatmap)
    controls_layout.addWidget(self.heatmap_type_combo)
    
    controls_layout.addStretch()
    heatmap_layout.addLayout(controls_layout)
    
    # Heatmap canvas
    self.heatmap_figure = Figure(figsize=(8, 6), dpi=100)
    self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
    heatmap_layout.addWidget(self.heatmap_canvas)
    
    self.heatmap_tab.setLayout(heatmap_layout)
    self.tabs.addTab(self.heatmap_tab, "Heatmap")

def update_heatmap(self):
    """Update the heatmap visualization."""
    time_period = self.heatmap_period_combo.currentText().lower()
    data_type = self.heatmap_type_combo.currentText()
    
    # Clear the figure
    self.heatmap_figure.clear()
    ax = self.heatmap_figure.add_subplot(111)
    
    # Get data based on type
    if data_type == "Visitor Count":
        zone_data = self.db.get_zone_activity(time_period)
        if not zone_data:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(zone_data, columns=['timestamp', 'zone_id', 'visitor_count', 'dwell_time_ms'])
        
        # Group by zone and sum visitor count
        zone_counts = df.groupby('zone_id')['visitor_count'].sum().reset_index()
        
        # Generate heatmap data
        zones = zone_counts['zone_id'].tolist()
        counts = zone_counts['visitor_count'].tolist()
        
        # Create bar chart
        ax.bar(zones, counts, color='skyblue')
        ax.set_title('Total Visitor Count by Zone')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Visitor Count')
        plt.xticks(rotation=45)
        
    elif data_type == "Dwell Time":
        zone_data = self.db.get_zone_activity(time_period)
        if not zone_data:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(zone_data, columns=['timestamp', 'zone_id', 'visitor_count', 'dwell_time_ms'])
        
        # Group by zone and average dwell time
        df['dwell_time_sec'] = df['dwell_time_ms'] / 1000
        zone_dwell = df.groupby('zone_id')['dwell_time_sec'].mean().reset_index()
        
        # Generate heatmap data
        zones = zone_dwell['zone_id'].tolist()
        dwell_times = zone_dwell['dwell_time_sec'].tolist()
        
        # Create bar chart
        ax.bar(zones, dwell_times, color='lightgreen')
        ax.set_title('Average Dwell Time by Zone')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Dwell Time (seconds)')
        plt.xticks(rotation=45)
        
    elif data_type == "Expression":
        expression_data = self.db.get_expression_data(time_period)
        if not expression_data:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(expression_data, columns=['timestamp', 'expression', 'count'])
        
        # Group by expression and sum counts
        expression_counts = df.groupby('expression')['count'].sum().reset_index()
        
        # Generate pie chart
        expressions = expression_counts['expression'].tolist()
        counts = expression_counts['count'].tolist()
        
        ax.pie(counts, labels=expressions, autopct='%1.1f%%', startangle=90)
        ax.set_title('Expression Distribution')
        ax.axis('equal')
    
    self.heatmap_figure.tight_layout()
    self.heatmap_canvas.draw()

    # Add to src/ui/tabs/analytics_tab.py

def init_ui(self):
    # Keep your existing init_ui code, then add:
    
    # Add visitor paths tab
    self.add_visitor_paths_tab()
    
    # Rest of your existing code...

def add_visitor_paths_tab(self):
    """Add a tab for visualizing visitor paths."""
    self.paths_tab = QWidget()
    paths_layout = QVBoxLayout()
    
    # Controls
    controls_layout = QHBoxLayout()
    
    # Time period selection
    controls_layout.addWidget(QLabel("Time Period:"))
    self.paths_period_combo = QComboBox()
    self.paths_period_combo.addItems(["Hour", "Day", "Week", "Month"])
    self.paths_period_combo.setCurrentIndex(1)  # Default to "Day"
    self.paths_period_combo.currentIndexChanged.connect(self.update_visitor_paths)
    controls_layout.addWidget(self.paths_period_combo)
    
    # Visualization type
    controls_layout.addWidget(QLabel("Visualization:"))
    self.paths_type_combo = QComboBox()
    self.paths_type_combo.addItems(["Heatmap", "Paths", "Both"])
    self.paths_type_combo.currentIndexChanged.connect(self.update_visitor_paths)
    controls_layout.addWidget(self.paths_type_combo)
    
    # Statistics
    self.paths_stats_label = QLabel("Unique Visitors: 0")
    controls_layout.addWidget(self.paths_stats_label)
    
    # Export button
    self.paths_export_btn = QPushButton("Export Paths")
    self.paths_export_btn.clicked.connect(self.export_visitor_paths)
    controls_layout.addWidget(self.paths_export_btn)
    
    controls_layout.addStretch()
    paths_layout.addLayout(controls_layout)
    
    # Visualization canvas
    self.paths_canvas = QLabel()
    self.paths_canvas.setMinimumSize(640, 480)
    self.paths_canvas.setAlignment(Qt.AlignCenter)
    self.paths_canvas.setStyleSheet("background-color: black;")
    paths_layout.addWidget(self.paths_canvas)
    
    self.paths_tab.setLayout(paths_layout)
    self.tabs.addTab(self.paths_tab, "Visitor Paths")

def update_visitor_paths(self):
    """Update the visitor paths visualization."""
    time_period = self.paths_period_combo.currentText().lower()
    visualization_type = self.paths_type_combo.currentText()
    
    # Update statistics
    unique_visitors = self.db.get_unique_visitors(time_period)
    self.paths_stats_label.setText(f"Unique Visitors: {unique_visitors}")
    
    # Get trajectory data
    trajectory_data = self.db.get_visitor_trajectories(time_period)
    if not trajectory_data:
        self.paths_canvas.setText("No visitor path data available")
        return
    
    # Create a blank image
    width, height = 640, 480  # Adjust based on your camera feed
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if visualization_type in ["Heatmap", "Both"]:
        # Create heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for _, _, trajectory_json in trajectory_data:
            try:
                trajectory = json.loads(trajectory_json)
                for x, y in trajectory:
                    if 0 <= x < width and 0 <= y < height:
                        # Add gaussian blob at each point
                        cv2.circle(heatmap, (int(x), int(y)), 15, 1.0, -1)
            except Exception as e:
                logger.error(f"Error creating heatmap: {e}")
        
        # Normalize and colorize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            if visualization_type == "Both":
                # Blend with original image
                alpha = 0.7
                image = cv2.addWeighted(heatmap_colored, alpha, image, 1-alpha, 0)
            else:
                image = heatmap_colored
    
    if visualization_type in ["Paths", "Both"]:
        # Draw individual paths
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255)   # Purple
        ]
        
        for _, visitor_id, trajectory_json in trajectory_data:
            try:
                trajectory = json.loads(trajectory_json)
                
                # Generate color based on visitor ID
                color_idx = int(visitor_id) % len(colors) if visitor_id.isdigit() else hash(visitor_id) % len(colors)
                color = colors[color_idx]
                
                # Convert trajectory to points
                points = []
                for point in trajectory:
                    x, y = point
                    if 0 <= x < width and 0 <= y < height:
                        points.append((int(x), int(y)))
                
                if len(points) > 1:
                    # Draw path
                    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, 2)
                    
                    # Draw start and end points
                    cv2.circle(image, points[0], 5, (0, 255, 0), -1)  # Green start
                    cv2.circle(image, points[-1], 5, (0, 0, 255), -1)  # Red end
                    
                    # Draw visitor ID at start point
                    cv2.putText(image, f"ID: {visitor_id}", (points[0][0] + 5, points[0][1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                logger.error(f"Error drawing path: {e}")
    
    # Convert to QImage and display
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    self.paths_canvas.setPixmap(QPixmap.fromImage(q_image))

def export_visitor_paths(self):
    """Export visitor paths data to CSV and image."""
    time_period = self.paths_period_combo.currentText().lower()
    
    # Ask for export directory
    export_dir = QFileDialog.getExistingDirectory(
        self, "Select Export Directory", "", QFileDialog.ShowDirsOnly
    )
    
    if not export_dir:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export paths image
    if self.paths_canvas.pixmap():
        image_path = os.path.join(export_dir, f"visitor_paths_{timestamp}.png")
        self.paths_canvas.pixmap().save(image_path)
        logger.info(f"Visitor paths image exported to {image_path}")
    
    # Export trajectory data
    trajectory_data = self.db.get_visitor_trajectories(time_period)
    if trajectory_data:
        # Create a flattened CSV format
        csv_path = os.path.join(export_dir, f"visitor_paths_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['timestamp', 'visitor_id', 'x', 'y'])
            
            # Write data
            for timestamp, visitor_id, trajectory_json in trajectory_data:
                try:
                    trajectory = json.loads(trajectory_json)
                    for x, y in trajectory:
                        csv_writer.writerow([timestamp, visitor_id, x, y])
                except Exception as e:
                    logger.error(f"Error exporting trajectory: {e}")
        
        logger.info(f"Visitor paths data exported to {csv_path}")
    
    QMessageBox.information(self, "Export Complete", 
                          f"Visitor paths data exported to {export_dir}")