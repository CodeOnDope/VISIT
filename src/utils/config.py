"""
Configuration management for the VISIT Museum Tracker system.

This module provides functionality for loading, saving, and accessing configuration
settings throughout the application.
"""

import os
import json
import logging


class Config:
    """Configuration manager for the VISIT Museum Tracker application."""
    
    def __init__(self, config_dir=None):
        """Initialize the configuration manager."""
        # Set up logger
        self.logger = logging.getLogger("Config")
        
        # Determine config directory
        if config_dir is None:
            # Get project root directory (assumed to be 2 levels up from this file)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            config_dir = os.path.join(project_root, "data", "settings")
        
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "config.json")
        
        # Initialize configuration dictionary
        self.config = {}
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Try to load configuration
        self.load()
    
    def load(self):
        """Load configuration from the config file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
            else:
                self.logger.info(f"Config file {self.config_file} does not exist, using defaults")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
    
    def save(self):
        """Save configuration to the config file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def load_defaults(self):
        """Load default configuration settings."""
        self.config = {
            "camera": {
                "camera_id": 0,
                "resolution": [640, 480],
                "fps": 30
            },
            "detectors": {
                "face": {
                    "enabled": True,
                    "min_detection_confidence": 0.5,
                    "model_selection": 0
                },
                "expression": {
                    "enabled": True,
                    "min_detection_confidence": 0.5,
                    "min_tracking_confidence": 0.5,
                    "max_num_faces": 1,
                    "enable_expressions": ["smile", "frown", "surprise", "neutral"]
                }
            },
            "ui": {
                "theme": "default",
                "video_resize_mode": "contain"
            },
            "analytics": {
                "enabled": True,
                "storage_path": "../data/analytics",
                "session_recording": False
            },
            "logging": {
                "level": "info",
                "file_logging": True,
                "log_directory": "../logs"
            }
        }
        self.logger.info("Loaded default configuration")
    
    def get(self, section, key=None, default=None):
        """Get a configuration value."""
        # If section doesn't exist, return default
        if section not in self.config:
            return default
        
        # If key is None, return the entire section
        if key is None:
            return self.config[section]
        
        # If section is a dictionary, get the key
        if isinstance(self.config[section], dict):
            return self.config[section].get(key, default)
        
        # If section is not a dictionary, return default
        return default
    
    def set(self, section, key, value):
        """Set a configuration value."""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def get_detector_config(self, detector_name):
        """Get configuration for a specific detector."""
        detectors = self.config.get("detectors", {})
        return detectors.get(detector_name, {})
    
    def set_detector_config(self, detector_name, config):
        """Set configuration for a specific detector."""
        if "detectors" not in self.config:
            self.config["detectors"] = {}
        
        self.config["detectors"][detector_name] = config
    
    def is_detector_enabled(self, detector_name):
        """Check if a detector is enabled in the configuration."""
        detector_config = self.get_detector_config(detector_name)
        return detector_config.get("enabled", False)