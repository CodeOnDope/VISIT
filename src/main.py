"""
Main entry point for the VISIT Museum Tracker application.

This module initializes and starts the VISIT Museum Tracker application.
"""

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from src.ui.main_window import MainWindow
from src.utils.logger import setup_application_logger


def main():
    """Main entry point for the application."""
    # Set up application logger
    logger = setup_application_logger()
    logger.info("Starting VISIT Museum Tracker application")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("VISIT Museum Tracker")
    app.setOrganizationName("VISIT")
    
    # Create and show main window
    window = MainWindow(app)
    window.show()
    
    # Run application event loop
    exit_code = app.exec_()
    
    # Log application exit
    logger.info(f"VISIT Museum Tracker application exited with code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
