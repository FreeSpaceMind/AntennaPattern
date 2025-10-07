"""
Worker thread for SWE calculations to prevent GUI freezing.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import logging

logger = logging.getLogger(__name__)


class SWEWorker(QThread):
    """Worker thread for calculating spherical wave expansion."""
    
    # Signals
    finished = pyqtSignal(dict)  # Emits swe_data when done
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(str)  # Emits progress messages
    
    def __init__(self, pattern, radius, frequency, adaptive):
        super().__init__()
        self.pattern = pattern
        self.radius = radius
        self.frequency = frequency
        self.adaptive = adaptive
    
    def run(self):
        """Run the calculation in background thread."""
        try:
            logger.info("SWE worker thread started")
            self.progress.emit("Calculating spherical modes...")
            
            swe_data = self.pattern.calculate_spherical_modes(
                radius=self.radius,
                frequency=self.frequency,
                adaptive=self.adaptive
            )
            
            logger.info("SWE calculation complete")
            self.finished.emit(swe_data)
            
        except Exception as e:
            logger.error(f"SWE calculation error: {e}", exc_info=True)
            self.error.emit(str(e))