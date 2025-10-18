"""
Worker thread for SWE calculations to prevent GUI freezing.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import logging

logger = logging.getLogger(__name__)


class SWEWorker(QThread):
    """Worker thread for calculating spherical wave expansion."""
    
    # Signals
    finished = pyqtSignal(object)  # Emits SWE object when done
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(str)  # Emits progress messages
    
    def __init__(self, pattern, frequency):
        super().__init__()
        self.pattern = pattern
        self.frequency = frequency

    def run(self):
        """Run the calculation in background thread."""
        try:
            logger.info("SWE worker thread started")
            self.progress.emit("Calculating spherical modes...")
            
            swe = self.pattern.calculate_spherical_modes(frequency=self.frequency)
            
            logger.info("SWE calculation complete")
            self.finished.emit(swe)
            
        except Exception as e:
            logger.error(f"SWE calculation error: {e}", exc_info=True)
            self.error.emit(str(e))