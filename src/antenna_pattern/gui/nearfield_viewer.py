"""
Near field viewer window for displaying near field plots.
File: src/antenna_pattern/gui/nearfield_viewer.py
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
                              QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np


class NearFieldViewer(QDialog):
    """Window for viewing near field plots."""
    
    def __init__(self, nf_data, parent=None):
        super().__init__(parent)
        self.nf_data = nf_data
        self.setWindowTitle("Near Field Viewer")
        self.resize(900, 700)
        self.setup_ui()
        self.plot_nearfield()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Controls at top
        controls_layout = QHBoxLayout()
        
        # Component selection
        controls_layout.addWidget(QLabel("Component:"))
        self.component_combo = QComboBox()
        # Will populate based on what's available in data
        available_components = self.get_available_components()
        self.component_combo.addItems(available_components)
        self.component_combo.currentTextChanged.connect(self.plot_nearfield)
        controls_layout.addWidget(self.component_combo)
        
        # Value type selection
        controls_layout.addWidget(QLabel("Value:"))
        self.value_combo = QComboBox()
        self.value_combo.addItems(["Magnitude (dBW)", "Phase (deg)", "Magnitude (V/m)"])
        self.value_combo.currentTextChanged.connect(self.plot_nearfield)
        controls_layout.addWidget(self.value_combo)
        
        controls_layout.addStretch()
        
        # Save button
        self.save_btn = QPushButton("Save Plot")
        self.save_btn.clicked.connect(self.save_plot)
        controls_layout.addWidget(self.save_btn)
        
        layout.addLayout(controls_layout)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def get_available_components(self):
        """Get list of available field components from data."""
        components = []
        
        # Check for different possible field component names
        possible_fields = [
            ('E_theta', 'E-theta'),
            ('E_phi', 'E-phi'),
            ('e_theta', 'E-theta'),
            ('e_phi', 'E-phi'),
            ('E_co', 'Co-pol'),
            ('E_cx', 'Cross-pol'),
            ('e_co', 'Co-pol'),
            ('e_cx', 'Cross-pol')
        ]
        
        for field_key, display_name in possible_fields:
            if field_key in self.nf_data:
                components.append(display_name)
        
        # If no components found, just list what's in the data
        if not components:
            for key in self.nf_data.keys():
                if key not in ['x', 'y', 'theta', 'phi', 'radius', 'is_spherical',
                               'x_extent', 'y_extent', 'z_distance']:
                    components.append(key)
        
        return components if components else ['E_theta']
    
    def get_field_data_key(self, display_name):
        """Map display name to actual data key."""
        name_map = {
            'E-theta': ['E_theta', 'e_theta'],
            'E-phi': ['E_phi', 'e_phi'],
            'Co-pol': ['E_co', 'e_co'],
            'Cross-pol': ['E_cx', 'e_cx']
        }
        
        # Try mapped names first
        if display_name in name_map:
            for key in name_map[display_name]:
                if key in self.nf_data:
                    return key
        
        # Try direct match
        if display_name in self.nf_data:
            return display_name
        
        # Return first available field
        for key in self.nf_data.keys():
            if key not in ['x', 'y', 'theta', 'phi', 'radius', 'is_spherical',
                           'x_extent', 'y_extent', 'z_distance']:
                return key
        
        return None
    
    def plot_nearfield(self):
        """Plot the near field data."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Get selected component and value type
        component_display = self.component_combo.currentText()
        value_type = self.value_combo.currentText()
        
        # Get the actual field data key
        field_key = self.get_field_data_key(component_display)
        
        if field_key is None or field_key not in self.nf_data:
            ax.text(0.5, 0.5, f'Field component not available',
                ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        field_data = self.nf_data[field_key]
        
        # Determine if this is an H field for proper labeling
        is_h_field = component_display.startswith('H-')
        
        # Calculate values based on type
        if "dB" in value_type:
            values = 20 * np.log10(np.abs(field_data) + 1e-12)
            label = 'Magnitude (dB)'
        elif "Phase" in value_type:
            values = np.angle(field_data, deg=True)
            label = 'Phase (degrees)'
        else:
            values = np.abs(field_data)
            label = 'Magnitude (A/m)' if is_h_field else 'Magnitude (V/m)'
        
        # Plot based on surface type
        if self.nf_data.get('is_spherical', True):
            # Spherical surface plot
            theta = self.nf_data['theta']
            phi = self.nf_data['phi']
            
            # Create meshgrid - match the indexing used in evaluate method
            THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
            
            # 2D color plot
            im = ax.pcolormesh(PHI, THETA, values, shading='auto', cmap='jet')
            ax.set_xlabel('Phi (degrees)', fontsize=11)
            ax.set_ylabel('Theta (degrees)', fontsize=11)
            ax.set_title(f'Near Field on Sphere - {component_display} - {label}', fontsize=12)
            
            cbar = self.figure.colorbar(im, ax=ax, label=label)
        else:
            # Planar surface plot
            x = self.nf_data['x']
            y = self.nf_data['y']
            
            # Create meshgrid - match the indexing used in evaluate method
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # 2D color plot - no transpose needed with 'ij' indexing
            im = ax.pcolormesh(X, Y, values, shading='auto', cmap='jet')
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_title(f'Near Field on Plane - {component_display} - {label}', fontsize=12)
            ax.set_aspect('equal')
            
            cbar = self.figure.colorbar(im, ax=ax, label=label)
        
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def save_plot(self):
        """Save the current plot to file."""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Near Field Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if filename:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')

    def get_available_components(self):
        """Get list of available field components from data."""
        components = []
        
        # Check for different possible field component names in order of preference
        possible_fields = [
            ('E_theta', 'E-theta'),
            ('E_phi', 'E-phi'),
            ('E_r', 'E-r'),
            ('E_x', 'E-x'),
            ('E_y', 'E-y'),
            ('E_z', 'E-z'),
            ('H_theta', 'H-theta'),
            ('H_phi', 'H-phi'),
            ('H_r', 'H-r'),
            ('H_x', 'H-x'),
            ('H_y', 'H-y'),
            ('H_z', 'H-z'),
            ('e_theta', 'E-theta'),
            ('e_phi', 'E-phi'),
            ('E_co', 'Co-pol'),
            ('E_cx', 'Cross-pol'),
            ('e_co', 'Co-pol'),
            ('e_cx', 'Cross-pol')
        ]
        
        for field_key, display_name in possible_fields:
            if field_key in self.nf_data:
                # Only add if not already in list (avoid duplicates)
                if display_name not in components:
                    components.append(display_name)
        
        # If no components found, just list what's in the data
        if not components:
            for key in self.nf_data.keys():
                if key not in ['x', 'y', 'theta', 'phi', 'radius', 'is_spherical',
                            'x_extent', 'y_extent', 'z_distance', 'frequency']:
                    components.append(key)
        
        return components if components else ['E_theta']
    
    def get_field_data_key(self, display_name):
        """Map display name to actual data key."""
        name_map = {
            'E-theta': ['E_theta', 'e_theta'],
            'E-phi': ['E_phi', 'e_phi'],
            'E-r': ['E_r', 'e_r'],
            'E-x': ['E_x', 'e_x'],
            'E-y': ['E_y', 'e_y'],
            'E-z': ['E_z', 'e_z'],
            'H-theta': ['H_theta', 'h_theta'],
            'H-phi': ['H_phi', 'h_phi'],
            'H-r': ['H_r', 'h_r'],
            'H-x': ['H_x', 'h_x'],
            'H-y': ['H_y', 'h_y'],
            'H-z': ['H_z', 'h_z'],
            'Co-pol': ['E_co', 'e_co'],
            'Cross-pol': ['E_cx', 'e_cx']
        }
        
        # Try mapped names first
        if display_name in name_map:
            for key in name_map[display_name]:
                if key in self.nf_data:
                    return key
        
        # Try direct match
        if display_name in self.nf_data:
            return display_name
        
        # Return first available field
        for key in self.nf_data.keys():
            if key not in ['x', 'y', 'theta', 'phi', 'radius', 'is_spherical',
                        'x_extent', 'y_extent', 'z_distance', 'frequency']:
                return key
        
        return None