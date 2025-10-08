#!/usr/bin/env python
"""
Entry point for the Antenna Pattern GUI application.
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from antenna_pattern.gui.main_window import MainWindow


def main():
    """Launch the GUI application."""
    
    # CONFIGURE LOGGING
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Antenna Pattern Analyzer GUI")
    
    # Enable high DPI display support
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Antenna Pattern Analyzer")
    app.setOrganizationName("AntennaPattern")
    
    # Create and show main window
    main_window = MainWindow()
    main_window.show()
    
    # Start the event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()