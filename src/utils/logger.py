 
"""
Logging utility for the VISIT Museum Tracker system.

This module provides a centralized logging setup for all components of the application.
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(name, level=logging.INFO, log_file=None):
    """Set up a logger with the specified name and level.
    
    Args:
        name (str): Logger name
        level (int, optional): Logging level. Defaults to logging.INFO.
        log_file (str, optional): Log file path. If None, a default log file is used.
            Defaults to None.
            
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # If log file is specified, add file handler
    if log_file is not None:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Add rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_application_logger(app_name="VISIT", level=logging.INFO):
    """Set up the main application logger.
    
    Args:
        app_name (str, optional): Application name. Defaults to "VISIT".
        level (int, optional): Logging level. Defaults to logging.INFO.
        
    Returns:
        logging.Logger: Configured application logger
    """
    # Get project root directory (assumed to be 2 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, "logs", "application")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{app_name}_{timestamp}.log")
    
    # Set up the logger
    logger = setup_logger(app_name, level=level, log_file=log_file)
    
    # Log startup message
    logger.info(f"=== {app_name} Application Started ===")
    
    return logger