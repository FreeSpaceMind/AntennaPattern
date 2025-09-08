"""
Matplotlib integration widget for PyQt6.
"""

import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for PyQt6 compatibility

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLineEdit, QLabel
from PyQt6.QtCore import pyqtSignal

from ..plotting import plot_pattern_cut


class PlotWidget(QWidget):
    """Widget containing matplotlib canvas and plot formatting controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the plot widget UI."""
        layout = QVBoxLayout()
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Plot formatting controls
        format_layout = QHBoxLayout()
        
        # Grid checkbox
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self.update_plot_formatting)
        format_layout.addWidget(self.grid_check)
        
        # Legend checkbox
        self.legend_check = QCheckBox("Legend")
        self.legend_check.setChecked(True)
        self.legend_check.toggled.connect(self.update_plot_formatting)
        format_layout.addWidget(self.legend_check)
        
        # Y-axis limits
        format_layout.addWidget(QLabel("Y-axis:"))
        self.y_min_edit = QLineEdit()
        self.y_min_edit.setPlaceholderText("Auto")
        self.y_min_edit.setMaximumWidth(80)
        self.y_min_edit.editingFinished.connect(self.update_plot_formatting)
        format_layout.addWidget(self.y_min_edit)
        
        format_layout.addWidget(QLabel("to"))
        self.y_max_edit = QLineEdit()
        self.y_max_edit.setPlaceholderText("Auto")
        self.y_max_edit.setMaximumWidth(80)
        self.y_max_edit.editingFinished.connect(self.update_plot_formatting)
        format_layout.addWidget(self.y_max_edit)
        
        format_layout.addStretch()
        
        # Add to main layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(format_layout)
        
        self.setLayout(layout)
    
    def update_plot(self, pattern, frequencies=None, phi_angles=None, 
                   polarization=None, value_type='gain', show_cross_pol=True):
        """Update the plot with new parameters."""
        if pattern is None:
            return
            
        self.current_pattern = pattern
        
        # Clear the figure
        self.figure.clear()
        
        # Create new axes
        ax = self.figure.add_subplot(111)
        
        try:
            # Use existing plot_pattern_cut function
            plot_pattern_cut(
                pattern=pattern,
                frequency=frequencies,
                phi=phi_angles,
                show_cross_pol=show_cross_pol,
                value_type=value_type,
                ax=ax,
                title=None  # Let the function generate title
            )
            
            # Apply formatting
            self.apply_plot_formatting(ax)
            
        except Exception as e:
            # Show error message on plot
            ax.text(0.5, 0.5, f"Plot Error:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Refresh canvas
        self.canvas.draw()
    
    def apply_plot_formatting(self, ax):
        """Apply current formatting settings to the axes."""
        # Grid
        ax.grid(self.grid_check.isChecked())
        
        # Legend
        if self.legend_check.isChecked():
            ax.legend(loc='best')
        else:
            legend = ax.get_legend()
            if legend:
                legend.remove()
        
        # Y-axis limits
        try:
            if self.y_min_edit.text().strip():
                y_min = float(self.y_min_edit.text())
            else:
                y_min = None
                
            if self.y_max_edit.text().strip():
                y_max = float(self.y_max_edit.text())
            else:
                y_max = None
                
            if y_min is not None or y_max is not None:
                current_limits = ax.get_ylim()
                new_min = y_min if y_min is not None else current_limits[0]
                new_max = y_max if y_max is not None else current_limits[1]
                ax.set_ylim(new_min, new_max)
                
        except ValueError:
            # Invalid input, ignore
            pass
    
    def update_plot_formatting(self):
        """Update plot formatting without replotting data."""
        if self.figure.axes:
            ax = self.figure.axes[0]
            self.apply_plot_formatting(ax)
            self.canvas.draw()
    
    def save_plot(self, filename):
        """Save the current plot to file."""
        self.figure.savefig(filename, dpi=300, bbox_inches='tight')
    
    def clear_plot(self):
        """Clear the current plot."""
        self.figure.clear()
        self.canvas.draw()
        self.current_pattern = None