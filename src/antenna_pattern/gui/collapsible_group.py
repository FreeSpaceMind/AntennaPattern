"""
Collapsible group box widget.
"""

from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QWidget, QToolButton
from PyQt6.QtCore import Qt


class CollapsibleGroupBox(QGroupBox):
    """A collapsible group box widget."""
    
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        
        # Create toggle button
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.toggle_collapsed)
        
        # Create content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(9, 0, 9, 9)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        
        # Start collapsed
        self.content_area.setVisible(False)
        
    def addWidget(self, widget):
        """Add widget to the content area."""
        self.content_layout.addWidget(widget)
        
    def addLayout(self, layout):
        """Add layout to the content area."""
        self.content_layout.addLayout(layout)
        
    def toggle_collapsed(self):
        """Toggle the collapsed state."""
        if self.content_area.isVisible():
            # Collapse
            self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self.content_area.setVisible(False)
            self.toggle_button.setChecked(False)
        else:
            # Expand  
            self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.content_area.setVisible(True)
            self.toggle_button.setChecked(True)