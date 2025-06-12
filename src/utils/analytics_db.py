# File: src/utils/analytics_db.py

import sqlite3
import os
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger("AnalyticsDB")

class AnalyticsDB:
    def __init__(self, db_path="data/analytics.db"):
        """Initialize the analytics database."""
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.initialize_db()

    def initialize_db(self):
        """Create database tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create visitor_count table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitor_trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            visitor_id TEXT NOT NULL,
            trajectory TEXT NOT NULL,
            camera_id TEXT DEFAULT 'default'
            )
            ''')
            
            # Create zone_activity table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS zone_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                visitor_count INTEGER NOT NULL,
                dwell_time_ms INTEGER DEFAULT 0,
                camera_id TEXT DEFAULT 'default'
            )
            ''')
            
            # Create expression_data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS expression_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                expression TEXT NOT NULL,
                count INTEGER NOT NULL,
                camera_id TEXT DEFAULT 'default'
            )
            ''')
            
            # Create visitor_paths table for tracking movement
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitor_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                visitor_id TEXT NOT NULL,
                position TEXT NOT NULL,
                camera_id TEXT DEFAULT 'default'
            )
            ''')
            
            self.conn.commit()
            logger.info("Analytics database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def add_visitor_count(self, count, camera_id='default'):
        """Record the current visitor count."""
        if not self.conn:
            self.initialize_db()
            
        try:
            timestamp = datetime.now().isoformat()
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO visitor_count (timestamp, count, camera_id) VALUES (?, ?, ?)",
                (timestamp, count, camera_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error recording visitor count: {e}")
    
    def add_zone_activity(self, zone_id, visitor_count, dwell_time_ms=0, camera_id='default'):
        """Record zone activity data."""
        if not self.conn:
            self.initialize_db()
            
        try:
            timestamp = datetime.now().isoformat()
            cursor = self.conn.cursor()
            cursor.execute(
            "INSERT INTO zone_activity (timestamp, zone_id, visitor_count, dwell_time_ms, camera_id) VALUES (?, ?, ?, ?, ?)",
                (timestamp, zone_id, visitor_count, dwell_time_ms, camera_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error recording zone activity: {e}")

    def add_expression_data(self, expression, count, camera_id='default'):
        """Record expression detection data."""
        if not self.conn:
            self.initialize_db()
            
        try:
            timestamp = datetime.now().isoformat()
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO expression_data (timestamp, expression, count, camera_id) VALUES (?, ?, ?, ?)",
                (timestamp, expression, count, camera_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error recording expression data: {e}")
    
    def add_visitor_path(self, visitor_id, position, camera_id='default'):
        """Record visitor movement paths.
        position should be a tuple or list of (x, y) coordinates."""
        if not self.conn:
            self.initialize_db()
            
        try:
            timestamp = datetime.now().isoformat()
            position_json = json.dumps(position)
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO visitor_paths (timestamp, visitor_id, position, camera_id) VALUES (?, ?, ?, ?)",
                (timestamp, visitor_id, position_json, camera_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error recording visitor path: {e}")
    
    def get_visitor_counts(self, time_period='day'):
        """Get visitor counts for the specified time period."""
        if not self.conn:
            self.initialize_db()
            
        try:
            now = datetime.now()
            if time_period == 'hour':
                start_time = (now - timedelta(hours=1)).isoformat()
            elif time_period == 'day':
                start_time = (now - timedelta(days=1)).isoformat()
            elif time_period == 'week':
                start_time = (now - timedelta(weeks=1)).isoformat()
            elif time_period == 'month':
                start_time = (now - timedelta(days=30)).isoformat()
            else:
                start_time = (now - timedelta(days=1)).isoformat()
                
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, count FROM visitor_count WHERE timestamp >= ? ORDER BY timestamp",
                (start_time,)
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving visitor counts: {e}")
            return []
    
    def get_zone_activity(self, time_period='day'):
        """Get zone activity data for the specified time period."""
        if not self.conn:
            self.initialize_db()
            
        try:
            now = datetime.now()
            if time_period == 'hour':
                start_time = (now - timedelta(hours=1)).isoformat()
            elif time_period == 'day':
                start_time = (now - timedelta(days=1)).isoformat()
            elif time_period == 'week':
                start_time = (now - timedelta(weeks=1)).isoformat()
            elif time_period == 'month':
                start_time = (now - timedelta(days=30)).isoformat()
            else:
                start_time = (now - timedelta(days=1)).isoformat()
                
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, zone_id, visitor_count, dwell_time_ms FROM zone_activity WHERE timestamp >= ? ORDER BY timestamp",
                (start_time,)
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving zone activity: {e}")
            return []
    
    def get_expression_data(self, time_period='day'):
        """Get expression data for the specified time period."""
        if not self.conn:
            self.initialize_db()
            
        try:
            now = datetime.now()
            if time_period == 'hour':
                start_time = (now - timedelta(hours=1)).isoformat()
            elif time_period == 'day':
                start_time = (now - timedelta(days=1)).isoformat()
            elif time_period == 'week':
                start_time = (now - timedelta(weeks=1)).isoformat()
            elif time_period == 'month':
                start_time = (now - timedelta(days=30)).isoformat()
            else:
                start_time = (now - timedelta(days=1)).isoformat()
                
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, expression, count FROM expression_data WHERE timestamp >= ? ORDER BY timestamp",
                (start_time,)
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving expression data: {e}")
            return []
            

    def add_visitor_trajectory(self, visitor_id, trajectory, camera_id='default'):
        """Record a visitor's movement trajectory."""
        if not self.conn:
            self.initialize_db()
        try:
            timestamp = datetime.now().isoformat()
            trajectory_json = json.dumps(trajectory)
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO visitor_trajectories (timestamp, visitor_id, trajectory, camera_id) VALUES (?, ?, ?, ?)",
                (timestamp, visitor_id, trajectory_json, camera_id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error recording visitor trajectory: {e}")

    def get_visitor_trajectories(self, time_period='day'):
        """Get visitor trajectory data for the specified time period."""
        if not self.conn:
            self.initialize_db()
        try:
            now = datetime.now()
            if time_period == 'hour':
                start_time = (now - timedelta(hours=1)).isoformat()
            elif time_period == 'day':
                start_time = (now - timedelta(days=1)).isoformat()
            elif time_period == 'week':
                start_time = (now - timedelta(weeks=1)).isoformat()
            elif time_period == 'month':
                start_time = (now - timedelta(days=30)).isoformat()
            else:
                start_time = (now - timedelta(days=1)).isoformat()

            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, visitor_id, trajectory FROM visitor_trajectories WHERE timestamp >= ? ORDER BY timestamp",
                (start_time,)
            )
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error retrieving visitor trajectories: {e}")
            return []

    def get_unique_visitors(self, time_period='day'):
        """Get count of unique visitors for the specified time period."""
        if not self.conn:
            self.initialize_db()
        try:
            now = datetime.now()
            if time_period == 'hour':
                start_time = (now - timedelta(hours=1)).isoformat()
            elif time_period == 'day':
                start_time = (now - timedelta(days=1)).isoformat()
            elif time_period == 'week':
                start_time = (now - timedelta(weeks=1)).isoformat()
            elif time_period == 'month':
                start_time = (now - timedelta(days=30)).isoformat()
            else:
                start_time = (now - timedelta(days=1)).isoformat()

            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT visitor_id) FROM visitor_trajectories WHERE timestamp >= ?",
                (start_time,)
            )
            result = cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"Error retrieving unique visitor count: {e}")
            return 0

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


            # Add to src/utils/analytics_db.py

# Inside the initialize_db method, add this table:


