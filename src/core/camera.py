"""Camera management for the VISIT system.""" 
 
import cv2 
import threading 
import time 
import numpy as np 
 
class Camera: 
    """Camera class for capturing video frames.""" 
 
    def __init__(self, camera_id=0, resolution=(640, 480)): 
        """Initialize the camera with ID and resolution.""" 
        self.camera_id = camera_id 
        self.resolution = resolution 
        self.capture = None 
        self.is_running = False 
        self.frame = None 
        self.lock = threading.Lock() 
        self.thread = None 
 
    def start(self): 
        """Start the camera capture.""" 
        if self.is_running: 
            return True 
 
        self.capture = cv2.VideoCapture(self.camera_id) 
        if not self.capture.isOpened(): 
            return False 
 
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0]) 
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1]) 
 
        self.is_running = True 
        self.thread = threading.Thread(target=self._update_frame) 
        self.thread.daemon = True 
        self.thread.start() 
 
        return True 
 
    def stop(self): 
        """Stop the camera capture.""" 
        self.is_running = False 
        if self.thread: 
            self.thread.join(timeout=1.0) 
        if self.capture: 
            self.capture.release() 
            self.capture = None 
 
    def _update_frame(self): 
        """Background thread to continuously capture frames.""" 
        while self.is_running: 
            ret, frame = self.capture.read() 
            if ret: 
                with self.lock: 
                    self.frame = frame 
            time.sleep(0.01)  # Small delay to reduce CPU usage 
 
    def get_frame(self): 
        """Get the latest frame from the camera.""" 
        with self.lock: 
            if self.frame is None: 
                return None 
            return self.frame.copy() 
 
    def get_properties(self): 
        """Get camera properties.""" 
        if not self.capture or not self.capture.isOpened(): 
            return {} 
 
        return { 
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
            "fps": self.capture.get(cv2.CAP_PROP_FPS), 
        } 
# In your src/core/camera.py file, ensure it emits frames properly:

def get_frame(self):
    """Get the current frame.
    
    Returns:
        numpy.ndarray: Current frame or None if no frame is available
    """
    if self.is_running and self.cap.isOpened():
        ret, frame = self.cap.read()
        if ret:
            return frame
    return None