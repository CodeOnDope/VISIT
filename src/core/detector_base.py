"""Base detector class for all VISIT detectors.""" 
 
from abc import ABC, abstractmethod 
import numpy as np 
 
class DetectorBase(ABC): 
    """Base class for all detectors in the VISIT system.""" 
 
    def __init__(self, config=None): 
        """Initialize the detector with optional configuration.""" 
        self.config = config or {} 
        self.is_running = False 
        self.results = None 
 
    @abstractmethod 
    def process_frame(self, frame): 
        """Process a single frame. Must be implemented by subclasses.""" 
        pass 
 
    def start(self): 
        """Start the detector.""" 
        self.is_running = True 
 
    def stop(self): 
        """Stop the detector.""" 
        self.is_running = False 
 
    def is_active(self): 
        """Check if the detector is currently active.""" 
        return self.is_running 
 
    def get_results(self): 
        """Get the latest detection results.""" 
        return self.results 
 
    def get_config(self): 
        """Get the current detector configuration.""" 
        return self.config 
 
    def set_config(self, config): 
        """Update the detector configuration.""" 
        self.config = config 
