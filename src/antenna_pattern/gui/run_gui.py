#!/usr/bin/env python
"""
Entry point for the Antenna Pattern GUI application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from antenna_pattern.gui.main_window import MainWindow


def main():
    """Launch the GUI application."""
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