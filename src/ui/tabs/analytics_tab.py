"""
Analytics tab for the VISIT Museum Tracker application.

This module implements the analytics tab UI that displays visitor analytics
and statistics based on detection data.
"""

import logging
import time
import csv
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt

from src.utils.logger import setup_logger
from src.utils.analytics_storage import AnalyticsStorage


class AnalyticsTab(QWidget):
    """Tab for displaying visitor analytics and statistics."""

    def __init__(self, parent=None):
        """Initialize the analytics tab.

        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)

        # Set up logging
        self.logger = setup_logger("AnalyticsTab", level=logging.INFO)

        # Initialize analytics storage
        self.storage = AnalyticsStorage()

        # Set up UI
        self._init_ui()

        # Load data initially
        self.load_data()

    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Create controls layout
        controls_layout = QHBoxLayout()

        # Time range selector
        controls_layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["Today", "This Week", "This Month", "Custom..."])
        controls_layout.addWidget(self.time_range_combo)

        # Data type selector
        controls_layout.addWidget(QLabel("Data Type:"))
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["Visitor Count", "Expressions", "Attention Time", "Movement"])
        controls_layout.addWidget(self.data_type_combo)

        # Export button
        self.export_button = QPushButton("Export Data")
        controls_layout.addWidget(self.export_button)

        # Add stretch to push controls to the left
        controls_layout.addStretch()

        # Add controls to main layout
        main_layout.addLayout(controls_layout)

        # Create tabs for different analytics views
        self.analytics_tabs = QTabWidget()

        # Dashboard tab
        dashboard_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_widget)

        # Summary group
        summary_group = QGroupBox("Summary")
        summary_layout = QHBoxLayout(summary_group)

        # Summary labels (will be updated dynamically)
        self.total_visitors_label = QLabel("Total Visitors: 0")
        self.avg_visit_time_label = QLabel("Average Visit Time: 0m 0s")
        self.positive_expressions_label = QLabel("Positive Expressions: 0%")

        summary_layout.addWidget(self.total_visitors_label)
        summary_layout.addWidget(self.avg_visit_time_label)
        summary_layout.addWidget(self.positive_expressions_label)

        dashboard_layout.addWidget(summary_group)

        # Placeholder label for charts
        chart_placeholder = QLabel("Charts will be displayed here\n(QtChart module required)")
        chart_placeholder.setAlignment(Qt.AlignCenter)
        chart_placeholder.setStyleSheet("background-color: #f0f0f0; padding: 20px;")
        chart_placeholder.setMinimumHeight(300)
        dashboard_layout.addWidget(chart_placeholder)

        self.analytics_tabs.addTab(dashboard_widget, "Dashboard")

        # Visitors tab
        visitors_widget = QWidget()
        visitors_layout = QVBoxLayout(visitors_widget)

        # Visitor data table
        self.visitor_table = QTableWidget(0, 3)
        self.visitor_table.setHorizontalHeaderLabels(["Date", "Count", "Duration"])
        visitors_layout.addWidget(self.visitor_table)

        self.analytics_tabs.addTab(visitors_widget, "Visitors")

        # Expressions tab
        expressions_widget = QWidget()
        expressions_layout = QVBoxLayout(expressions_widget)

        # Placeholder expression data table
        self.expression_table = QTableWidget(5, 2)
        self.expression_table.setHorizontalHeaderLabels(["Expression", "Percentage"])
        expressions_layout.addWidget(self.expression_table)

        self.analytics_tabs.addTab(expressions_widget, "Expressions")

        # Movement tab
        movement_widget = QWidget()
        movement_layout = QVBoxLayout(movement_widget)

        # Placeholder movement analytics label
        movement_layout.addWidget(QLabel("Movement analytics will be displayed here."))

        self.analytics_tabs.addTab(movement_widget, "Movement")

        # Export tab
        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)

        # Placeholder export controls label
        export_layout.addWidget(QLabel("Data export controls will be available here."))

        self.analytics_tabs.addTab(export_widget, "Export")

        # Add tabs to main layout
        main_layout.addWidget(self.analytics_tabs)

        # Connect signals
        self.time_range_combo.currentIndexChanged.connect(self.load_data)
        self.export_button.clicked.connect(self.export_data)

    def load_data(self):
        """Load and display analytics data based on current filters."""
        try:
            now = time.time()
            range_text = self.time_range_combo.currentText()

            if range_text == "Today":
                since = now - 86400  # 24 hours
            elif range_text == "This Week":
                since = now - 7 * 86400
            elif range_text == "This Month":
                since = now - 30 * 86400
            else:
                since = None  # Custom range not implemented

            detections = self.storage.query_detections(since=since)

            # Total visitor count (approximate count of detection events)
            visitor_count = len(detections)

            # Placeholder average visit time and positive expressions (could be computed)
            avg_visit_time = "N/A"
            positive_expr_pct = "N/A"

            # Update summary labels
            self.total_visitors_label.setText(f"Total Visitors: {visitor_count}")
            self.avg_visit_time_label.setText(f"Average Visit Time: {avg_visit_time}")
            self.positive_expressions_label.setText(f"Positive Expressions: {positive_expr_pct}")

            # Aggregate visitor counts by date
            date_counts = {}
            for timestamp, detector, _, _, _, _ in detections:
                day_str = time.strftime("%Y-%m-%d", time.localtime(timestamp))
                date_counts[day_str] = date_counts.get(day_str, 0) + 1

            # Populate visitor table
            self.visitor_table.setRowCount(len(date_counts))
            for row, (date_str, count) in enumerate(sorted(date_counts.items())):
                self.visitor_table.setItem(row, 0, QTableWidgetItem(date_str))
                self.visitor_table.setItem(row, 1, QTableWidgetItem(str(count)))
                self.visitor_table.setItem(row, 2, QTableWidgetItem("N/A"))  # Duration placeholder

        except Exception as e:
            self.logger.error(f"Failed to load analytics data: {e}")

    def export_data(self):
        """Export visitor table data to CSV file."""
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Export Visitor Data", "", "CSV Files (*.csv)")
            if not path:
                return

            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = ["Date", "Count", "Duration"]
                writer.writerow(headers)

                for row in range(self.visitor_table.rowCount()):
                    row_data = []
                    for col in range(self.visitor_table.columnCount()):
                        item = self.visitor_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

            QMessageBox.information(self, "Export Successful", f"Data exported to {path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting data:\n{e}")

    def update_tab(self):
        """Update the tab content."""
        # This can be expanded for dynamic updates if needed
        pass

    def on_tab_selected(self):
        """Handle tab selected event."""
        # This can be expanded to trigger updates on tab change if needed
        pass
