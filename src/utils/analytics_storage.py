import sqlite3
import threading
import time
from typing import List, Tuple

class AnalyticsStorage:
    def __init__(self, db_path='data/analytics/detections.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    detector TEXT,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER
                )
            """)
            self.conn.commit()

    def insert_detection(self, detector: str, bbox: Tuple[int, int, int, int], timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO detections (timestamp, detector, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, detector, bbox[0], bbox[1], bbox[2], bbox[3]))
            self.conn.commit()

    def query_detections(self, detector: str = None, since: float = None) -> List[Tuple]:
        query = "SELECT timestamp, detector, bbox_x, bbox_y, bbox_w, bbox_h FROM detections WHERE 1=1"
        params = []
        if detector:
            query += " AND detector = ?"
            params.append(detector)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def close(self):
        with self.lock:
            self.conn.close()
