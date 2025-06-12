import os
import time
import threading
import logging
from PyQt5.QtCore import QTimer

logger = logging.getLogger("MediaController")

class MediaController:
    def __init__(self, media_folder="media/"):
        self.media_folder = media_folder
        self.media_queue = []  # List to hold queued media
        self.current_media = None
        self.is_playing = False
        self.media_timer = QTimer()
        self.media_timer.timeout.connect(self._check_media_status)
        self.media_timer.start(1000)  # check every second
    
    def start_playback(self):
        """Start playing media from the queue."""
        if not self.is_playing and self.media_queue:
            self.is_playing = True
            self.current_media = self.media_queue.pop(0)
            logger.info(f"Started playing: {self.current_media}")
            # Actual code to start the media, e.g., using a media player API or system call
            self._play_media(self.current_media)

    def _play_media(self, media_name):
        """Play the media file (dummy function for now)."""
        media_path = os.path.join(self.media_folder, media_name)
        logger.info(f"Playing media: {media_path}")
        # Dummy sleep to simulate media playing time
        time.sleep(5)  # Simulate 5 seconds of media playtime
        self.is_playing = False
        self._advance_media()  # Move to the next media in the queue
    
    def pause_playback(self):
        """Pause the current media."""
        if self.is_playing:
            self.is_playing = False
            logger.info("Paused media playback.")
            # Implement the actual pause functionality with your media player

    def resume_playback(self):
        """Resume playing the paused media."""
        if not self.is_playing and self.current_media:
            self.is_playing = True
            logger.info("Resumed media playback.")
            # Implement the actual resume functionality with your media player
    
    def _advance_media(self):
        """Advance to the next media in the queue."""
        if self.media_queue:
            self.start_playback()
        else:
            logger.info("No more media in the queue.")

    def trigger_media(self, media_name):
        """Add media to the queue."""
        if media_name not in self.media_queue:
            self.media_queue.append(media_name)
            logger.info(f"Media {media_name} added to queue.")
            if not self.is_playing:
                self.start_playback()
    
    def _check_media_status(self):
        """Check the status of the current media and move to next if needed."""
        if not self.is_playing and self.current_media:
            logger.info(f"Finished playing {self.current_media}.")
            self._advance_media()
