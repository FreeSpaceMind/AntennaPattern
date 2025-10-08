"""
Matplotlib integration widget for PyQt6.
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for PyQt6 compatibility

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLineEdit, QLabel
from PyQt6.QtCore import pyqtSignal

from ..plotting import plot_pattern_cut, plot_pattern_2d_polar


class PlotWidget(QWidget):
    """Widget containing matplotlib canvas and plot formatting controls."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        
        # Store axis limits for each plot type
        self.axis_limits_memory = {
            '1d_cut': {
                'x_min': '', 'x_max': '', 
                'y_min': '', 'y_max': ''
            },
            '2d_polar': {
                'phi_min': '', 'phi_max': '',     # Angular limits
                'theta_min': '', 'theta_max': '',  # Radial limits  
                'z_min': '', 'z_max': ''          # Colorbar limits
            }
        }
        
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
        
        # Legend/Colorbar checkbox
        self.legend_colorbar_check = QCheckBox("Legend")
        self.legend_colorbar_check.setChecked(True)
        self.legend_colorbar_check.toggled.connect(self.update_plot_formatting)
        format_layout.addWidget(self.legend_colorbar_check)

        # Normalize Checkbox
        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(False)
        self.normalize_check.toggled.connect(self.replot_current_data)
        format_layout.addWidget(self.normalize_check)
        
        # X-axis/Phi limits
        self.x_phi_label = QLabel("X-axis:")
        format_layout.addWidget(self.x_phi_label)
        
        self.x_phi_min_edit = QLineEdit()
        self.x_phi_min_edit.setPlaceholderText("Auto")
        self.x_phi_min_edit.setMaximumWidth(60)
        self.x_phi_min_edit.editingFinished.connect(self.update_plot_formatting)
        format_layout.addWidget(self.x_phi_min_edit)
        
        format_layout.addWidget(QLabel("to"))
        
        self.x_phi_max_edit = QLineEdit()
        self.x_phi_max_edit.setPlaceholderText("Auto")
        self.x_phi_max_edit.setMaximumWidth(60)
        self.x_phi_max_edit.editingFinished.connect(self.update_plot_formatting)
        format_layout.addWidget(self.x_phi_max_edit)
        
        # Y-axis/Theta limits
        self.y_theta_label = QLabel("Y-axis:")
        format_layout.addWidget(self.y_theta_label)
        
        self.y_theta_min_edit = QLineEdit()
        self.y_theta_min_edit.setPlaceholderText("Auto")
        self.y_theta_min_edit.setMaximumWidth(60)
        self.y_theta_min_edit.editingFinished.connect(self.update_plot_formatting)
        format_layout.addWidget(self.y_theta_min_edit)
        
        format_layout.addWidget(QLabel("to"))
        
        self.y_theta_max_edit = QLineEdit()
        self.y_theta_max_edit.setPlaceholderText("Auto")
        self.y_theta_max_edit.setMaximumWidth(60)
        self.y_theta_max_edit.editingFinished.connect(self.update_plot_formatting)
        format_layout.addWidget(self.y_theta_max_edit)
        
        # Z-axis/Colorbar limits (for 2D plots only)
        self.z_label = QLabel("Z-axis:")
        self.z_label.setVisible(False)
        format_layout.addWidget(self.z_label)
        
        self.z_min_edit = QLineEdit()
        self.z_min_edit.setPlaceholderText("Auto")
        self.z_min_edit.setMaximumWidth(60)
        self.z_min_edit.editingFinished.connect(self.update_plot_formatting)
        self.z_min_edit.setVisible(False)
        format_layout.addWidget(self.z_min_edit)
        
        self.z_to_label = QLabel("to")
        self.z_to_label.setVisible(False)
        format_layout.addWidget(self.z_to_label)
        
        self.z_max_edit = QLineEdit()
        self.z_max_edit.setPlaceholderText("Auto")
        self.z_max_edit.setMaximumWidth(60)
        self.z_max_edit.editingFinished.connect(self.update_plot_formatting)
        self.z_max_edit.setVisible(False)
        format_layout.addWidget(self.z_max_edit)
        
        format_layout.addStretch()
        
        # Add to main layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addLayout(format_layout)
        
        self.setLayout(layout)
        
        # Store current plot format for control updates
        self.current_plot_format = '1d_cut'
        self.current_colorbar = None

    
    def update_plot(self, pattern, frequencies, phi_angles, value_type, 
                    show_cross_pol, plot_format, component, 
                    statistics_enabled=False, show_range=True, 
                    statistic_type='mean', percentile_range=(25, 75)
    ):
        """
        Update the plot with new data and parameters.
        
        Args:
            pattern: AntennaPattern object
            frequencies: Frequency or list of frequencies to plot
            phi_angles: Phi angle or list of phi angles to plot
            value_type: Type of value to plot ('gain', 'phase', 'axial_ratio')
            show_cross_pol: Whether to show cross-polarization
            plot_format: Plot format ('1d_cut', '2d_polar', 'near_field')
            component: Component to plot ('e_co', 'e_cx', 'e_theta', 'e_phi')
            statistics_enabled: Whether to plot statistics instead of individual cuts
            show_range: Whether to show min/max range for statistics
            statistic_type: Type of statistic ('mean', 'median', 'rms', 'percentile', 'std')
            percentile_range: Tuple of (lower, upper) percentiles
        """
        import numpy as np
        from ..plotting import plot_pattern_cut, plot_pattern_2d_polar, plot_pattern_statistics
        
        # Store current parameters for formatting updates
        self.current_pattern = pattern
        self.current_frequencies = frequencies
        self.current_phi_angles = phi_angles
        self.current_value_type = value_type
        self.current_show_cross_pol = show_cross_pol
        self.current_plot_format = plot_format
        self.current_component = component
        self.current_statistics_enabled = statistics_enabled
        self.current_show_range = show_range
        self.current_statistic_type = statistic_type
        self.current_percentile_range = percentile_range
        
        # Clear the current figure
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        try:
            # Statistics plot
            if statistics_enabled:
                # Determine statistic_over based on what's selected
                if isinstance(phi_angles, list) and len(phi_angles) > 1:
                    statistic_over = 'phi'
                    freq_for_stats = frequencies if isinstance(frequencies, (int, float)) else frequencies[0]
                elif isinstance(frequencies, list) and len(frequencies) > 1:
                    statistic_over = 'frequency'
                    freq_for_stats = None
                else:
                    # Default to phi if only one of each is selected
                    statistic_over = 'phi'
                    freq_for_stats = frequencies if isinstance(frequencies, (int, float)) else frequencies[0]
                
                phi_for_stats = None if statistic_over == 'phi' else (
                    phi_angles if isinstance(phi_angles, (int, float)) else phi_angles[0]
                )
                
                plot_pattern_statistics(
                    pattern=pattern,
                    statistic_over=statistic_over,
                    frequency=freq_for_stats,
                    phi=phi_for_stats,
                    component=component,
                    value_type=value_type,
                    statistic=statistic_type,
                    percentile_range=percentile_range,
                    show_range=show_range,
                    ax=self.ax
                )
            
            # 2D polar plot
            elif plot_format == "2d_polar":
                plot_pattern_2d_polar(
                    pattern=pattern,
                    frequency=frequencies,
                    component=component,
                    value_type=value_type,
                    ax=self.ax
                )
            
            # 1D cut plot (default)
            else:
                plot_pattern_cut(
                    pattern=pattern,
                    frequency=frequencies,
                    phi=phi_angles,
                    show_cross_pol=show_cross_pol,
                    value_type=value_type,
                    component=component,
                    ax=self.ax
                )
            
            # Apply formatting
            self.update_plot_formatting()
            
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error plotting:\n{str(e)}',
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=10, color='red')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis('off')
            print(f"Plotting error: {e}")
            import traceback
            traceback.print_exc()
        
        self.canvas.draw()

    def update_controls_for_plot_format(self):
        """Update axis control visibility and memory based on current plot format in PlotWidget."""
        # Use the actual plot format that was set (this comes from the plot_format parameter)
        is_2d = (self.current_plot_format == '2d_polar')
        
        if is_2d:
            # Save current 1D axis limits before switching
            if hasattr(self, 'axis_limits_memory'):
                self.save_current_axis_limits('1d_cut')
            
            # Update labels for 2D polar plot
            self.legend_colorbar_check.setText("Colorbar")
            self.y_theta_label.setText("Theta:")
            
            # Hide X-axis controls for 2D plots
            self.x_phi_label.setVisible(False)
            self.x_phi_min_edit.setVisible(False)
            self.x_phi_max_edit.setVisible(False)
            
            # Find and hide the "to" label between X controls
            parent_layout = self.x_phi_min_edit.parent().layout()
            for i in range(parent_layout.count()):
                item = parent_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    widget = item.widget()
                    if (widget.text() == "to" and 
                        widget != self.z_to_label and 
                        # Check if it's between the X controls
                        parent_layout.indexOf(widget) > parent_layout.indexOf(self.x_phi_min_edit) and
                        parent_layout.indexOf(widget) < parent_layout.indexOf(self.y_theta_min_edit)):
                        widget.setVisible(False)
                        break
            
            # Show Z-axis controls for colorbar limits
            self.z_label.setVisible(True)
            self.z_min_edit.setVisible(True)
            self.z_to_label.setVisible(True)
            self.z_max_edit.setVisible(True)
            
            # Restore 2D axis limits
            if hasattr(self, 'axis_limits_memory'):
                self.restore_axis_limits('2d_polar')
            
        else:
            # Save current 2D axis limits before switching
            if hasattr(self, 'axis_limits_memory'):
                self.save_current_axis_limits('2d_polar')
            
            # Update labels for 1D cut plot
            self.legend_colorbar_check.setText("Legend")
            self.x_phi_label.setText("X-axis:")
            self.y_theta_label.setText("Y-axis:")
            
            # Show X-axis controls for 1D plots
            self.x_phi_label.setVisible(True)
            self.x_phi_min_edit.setVisible(True)
            self.x_phi_max_edit.setVisible(True)
            
            # Find and show the "to" label between X controls
            parent_layout = self.x_phi_min_edit.parent().layout()
            for i in range(parent_layout.count()):
                item = parent_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    widget = item.widget()
                    if (widget.text() == "to" and 
                        widget != self.z_to_label and 
                        # Check if it's between the X controls
                        parent_layout.indexOf(widget) > parent_layout.indexOf(self.x_phi_min_edit) and
                        parent_layout.indexOf(widget) < parent_layout.indexOf(self.y_theta_min_edit)):
                        widget.setVisible(True)
                        break
            
            # Hide Z-axis controls for 1D plots
            self.z_label.setVisible(False)
            self.z_min_edit.setVisible(False)
            self.z_to_label.setVisible(False)
            self.z_max_edit.setVisible(False)
            
            # Restore 1D axis limits
            if hasattr(self, 'axis_limits_memory'):
                self.restore_axis_limits('1d_cut')

    def get_colorbar_limits(self):
        """Get colorbar limits from Z-axis controls."""
        try:
            vmin = float(self.z_min_edit.text()) if self.z_min_edit.text().strip() else None
        except ValueError:
            vmin = None
            
        try:
            vmax = float(self.z_max_edit.text()) if self.z_max_edit.text().strip() else None
        except ValueError:
            vmax = None
            
        return vmin, vmax

    def get_colorbar_limits(self):
        """Get colorbar limits from Z-axis controls."""
        try:
            vmin = float(self.z_min_edit.text()) if self.z_min_edit.text().strip() else None
        except ValueError:
            vmin = None
            
        try:
            vmax = float(self.z_max_edit.text()) if self.z_max_edit.text().strip() else None
        except ValueError:
            vmax = None
            
        return vmin, vmax
    
    def apply_plot_formatting(self, ax):
        """Apply current formatting settings to the axes."""
        # Check if this is a polar plot
        is_polar = hasattr(ax, 'set_theta_zero_location')
        
        # Grid - works for both regular and polar plots
        ax.grid(self.grid_check.isChecked())
        
        if is_polar:
            # Handle polar plot formatting
            
            # Colorbar visibility
            if hasattr(self, 'current_colorbar') and self.current_colorbar:
                self.current_colorbar.ax.set_visible(self.legend_colorbar_check.isChecked())
                
                # Update colorbar limits if changed
                try:
                    vmin, vmax = self.get_colorbar_limits()
                    if vmin is not None or vmax is not None:
                        mappable = self.current_colorbar.mappable
                        current_vmin, current_vmax = mappable.get_clim()
                        new_vmin = vmin if vmin is not None else current_vmin
                        new_vmax = vmax if vmax is not None else current_vmax
                        mappable.set_clim(vmin=new_vmin, vmax=new_vmax)
                        self.current_colorbar.update_normal(mappable)
                except (ValueError, AttributeError):
                    pass
            
            # Phi (angular) limits - only if X-axis controls are visible
            # (Skip this since we're hiding X-axis controls for 2D plots)
            # Users can still zoom/pan with toolbar if needed
            
            # Theta (radial) limits
            try:
                theta_min = float(self.y_theta_min_edit.text()) if self.y_theta_min_edit.text().strip() else None
                theta_max = float(self.y_theta_max_edit.text()) if self.y_theta_max_edit.text().strip() else None
                
                if theta_min is not None or theta_max is not None:
                    current_limits = ax.get_ylim()
                    new_min = theta_min if theta_min is not None else current_limits[0]
                    new_max = theta_max if theta_max is not None else current_limits[1]
                    ax.set_ylim(max(0, new_min), new_max)
            except ValueError:
                pass
                
        else:
            # Handle regular plot formatting
            
            # Legend for 1D plots
            if self.legend_colorbar_check.isChecked():
                ax.legend(loc='best')
            else:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
            
            # X-axis limits for 1D plots
            try:
                x_min = float(self.x_phi_min_edit.text()) if self.x_phi_min_edit.text().strip() else None
                x_max = float(self.x_phi_max_edit.text()) if self.x_phi_max_edit.text().strip() else None
                
                if x_min is not None or x_max is not None:
                    current_limits = ax.get_xlim()
                    new_min = x_min if x_min is not None else current_limits[0]
                    new_max = x_max if x_max is not None else current_limits[1]
                    ax.set_xlim(new_min, new_max)
            except ValueError:
                pass
            
            # Y-axis limits for 1D plots
            try:
                y_min = float(self.y_theta_min_edit.text()) if self.y_theta_min_edit.text().strip() else None
                y_max = float(self.y_theta_max_edit.text()) if self.y_theta_max_edit.text().strip() else None
                
                if y_min is not None or y_max is not None:
                    current_limits = ax.get_ylim()
                    new_min = y_min if y_min is not None else current_limits[0]
                    new_max = y_max if y_max is not None else current_limits[1]
                    ax.set_ylim(new_min, new_max)
            except ValueError:
                pass
    def replot_current_data(self):
        """Replot using stored parameters."""
        if self.current_pattern is not None:
            self.update_plot(
                pattern=self.current_pattern,
                frequencies=self.current_frequencies,
                phi_angles=self.current_phi_angles,
                value_type=self.current_value_type,
                show_cross_pol=self.current_show_cross_pol,
                plot_format=self.current_plot_format,
                component=self.current_component,
                statistics_enabled=self.current_statistics_enabled,
                show_range=self.current_show_range,
                statistic_type=self.current_statistic_type,
                percentile_range=self.current_percentile_range
            )
            
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

    def save_current_axis_limits(self, plot_type):
        """Save current axis limits for the specified plot type."""
        if plot_type == '1d_cut':
            self.axis_limits_memory['1d_cut']['x_min'] = self.x_phi_min_edit.text()
            self.axis_limits_memory['1d_cut']['x_max'] = self.x_phi_max_edit.text()
            self.axis_limits_memory['1d_cut']['y_min'] = self.y_theta_min_edit.text()
            self.axis_limits_memory['1d_cut']['y_max'] = self.y_theta_max_edit.text()
            
        elif plot_type == '2d_polar':
            self.axis_limits_memory['2d_polar']['phi_min'] = self.x_phi_min_edit.text()
            self.axis_limits_memory['2d_polar']['phi_max'] = self.x_phi_max_edit.text()
            self.axis_limits_memory['2d_polar']['theta_min'] = self.y_theta_min_edit.text()
            self.axis_limits_memory['2d_polar']['theta_max'] = self.y_theta_max_edit.text()
            self.axis_limits_memory['2d_polar']['z_min'] = self.z_min_edit.text()
            self.axis_limits_memory['2d_polar']['z_max'] = self.z_max_edit.text()

    def restore_axis_limits(self, plot_type):
        """Restore axis limits for the specified plot type."""
        if plot_type == '1d_cut':
            limits = self.axis_limits_memory['1d_cut']
            self.x_phi_min_edit.setText(limits['x_min'])
            self.x_phi_max_edit.setText(limits['x_max'])
            self.y_theta_min_edit.setText(limits['y_min'])
            self.y_theta_max_edit.setText(limits['y_max'])
            
        elif plot_type == '2d_polar':
            limits = self.axis_limits_memory['2d_polar']
            self.x_phi_min_edit.setText(limits['phi_min'])
            self.x_phi_max_edit.setText(limits['phi_max'])
            self.y_theta_min_edit.setText(limits['theta_min'])
            self.y_theta_max_edit.setText(limits['theta_max'])
            self.z_min_edit.setText(limits['z_min'])
            self.z_max_edit.setText(limits['z_max'])

    def clear_axis_limits(self, plot_type=None):
        """Clear axis limits for specified plot type or all types."""
        if plot_type is None:
            # Clear all
            for ptype in self.axis_limits_memory:
                for key in self.axis_limits_memory[ptype]:
                    self.axis_limits_memory[ptype][key] = ''
        else:
            # Clear specific plot type
            if plot_type in self.axis_limits_memory:
                for key in self.axis_limits_memory[plot_type]:
                    self.axis_limits_memory[plot_type][key] = ''
        
        # Also clear current UI
        self.x_phi_min_edit.setText('')
        self.x_phi_max_edit.setText('')
        self.y_theta_min_edit.setText('')
        self.y_theta_max_edit.setText('')
        self.z_min_edit.setText('')
        self.z_max_edit.setText('')