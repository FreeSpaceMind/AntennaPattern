"""
Analysis tab - Controls for computing derived quantities from patterns.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QComboBox, QCheckBox, QPushButton, QDoubleSpinBox,
                            QScrollArea, QSpinBox, QTextEdit)
from PyQt6.QtCore import pyqtSignal, Qt

from .collapsible_group import CollapsibleGroupBox


class AnalysisTab(QWidget):
    """Tab containing analysis and computation controls."""
    
    # Signals for analysis operations
    calculate_swe_signal = pyqtSignal()
    calculate_nearfield_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        self.swe_calculated = False
        self.nearfield_data = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the analysis tab UI."""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Create scrollable widget
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Spherical Wave Expansion section
        swe_group = CollapsibleGroupBox("Spherical Wave Expansion")
        
        swe_group.addWidget(QLabel("Calculate spherical mode coefficients:"))
        
        # Adaptive calculation checkbox
        self.swe_adaptive_check = QCheckBox("Adaptive (auto-determine radius)")
        self.swe_adaptive_check.setChecked(True)
        self.swe_adaptive_check.toggled.connect(self.on_swe_adaptive_toggled)
        swe_group.addWidget(self.swe_adaptive_check)
        
        # Initial radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Initial Radius:"))
        self.swe_radius_spin = QDoubleSpinBox()
        self.swe_radius_spin.setRange(0.001, 10.0)
        self.swe_radius_spin.setValue(0.1)
        self.swe_radius_spin.setSuffix(" m")
        self.swe_radius_spin.setDecimals(4)
        self.swe_radius_spin.setEnabled(False)  # Disabled for adaptive
        radius_layout.addWidget(self.swe_radius_spin)
        swe_group.addLayout(radius_layout)
        
        # Frequency selection for SWE
        swe_freq_layout = QHBoxLayout()
        swe_freq_layout.addWidget(QLabel("Frequency:"))
        self.swe_freq_combo = QComboBox()
        swe_freq_layout.addWidget(self.swe_freq_combo)
        swe_group.addLayout(swe_freq_layout)
        
        # Calculate button
        self.calculate_swe_btn = QPushButton("Calculate SWE Coefficients")
        self.calculate_swe_btn.clicked.connect(self.on_calculate_swe)
        swe_group.addWidget(self.calculate_swe_btn)
        
        # Results display
        self.swe_results = QTextEdit()
        self.swe_results.setReadOnly(True)
        self.swe_results.setMaximumHeight(100)
        self.swe_results.setPlaceholderText("SWE results will appear here...")
        swe_group.addWidget(self.swe_results)
        
        layout.addWidget(swe_group)
        
        # Near Field Evaluation section
        nf_group = CollapsibleGroupBox("Near Field Evaluation")
        
        nf_group.addWidget(QLabel("Evaluate near field from SWE coefficients:"))
        
        # Surface type selection
        surface_layout = QHBoxLayout()
        surface_layout.addWidget(QLabel("Surface Type:"))
        self.nf_surface_combo = QComboBox()
        self.nf_surface_combo.addItems(["Spherical Surface", "Planar Surface"])
        self.nf_surface_combo.currentTextChanged.connect(self.on_surface_type_changed)
        surface_layout.addWidget(self.nf_surface_combo)
        nf_group.addLayout(surface_layout)
        
        # Spherical surface controls
        self.sphere_controls_widget = QWidget()
        sphere_layout = QVBoxLayout(self.sphere_controls_widget)
        sphere_layout.setContentsMargins(0, 0, 0, 0)
        
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        self.nf_sphere_radius_spin = QDoubleSpinBox()
        self.nf_sphere_radius_spin.setRange(0.001, 10.0)
        self.nf_sphere_radius_spin.setValue(0.05)
        self.nf_sphere_radius_spin.setSuffix(" m")
        self.nf_sphere_radius_spin.setDecimals(4)
        radius_layout.addWidget(self.nf_sphere_radius_spin)
        sphere_layout.addLayout(radius_layout)
        
        theta_pts_layout = QHBoxLayout()
        theta_pts_layout.addWidget(QLabel("Theta Points:"))
        self.nf_theta_points_spin = QSpinBox()
        self.nf_theta_points_spin.setRange(10, 361)
        self.nf_theta_points_spin.setValue(91)
        theta_pts_layout.addWidget(self.nf_theta_points_spin)
        sphere_layout.addLayout(theta_pts_layout)
        
        phi_pts_layout = QHBoxLayout()
        phi_pts_layout.addWidget(QLabel("Phi Points:"))
        self.nf_phi_points_spin = QSpinBox()
        self.nf_phi_points_spin.setRange(10, 361)
        self.nf_phi_points_spin.setValue(91)
        phi_pts_layout.addWidget(self.nf_phi_points_spin)
        sphere_layout.addLayout(phi_pts_layout)
        
        nf_group.addWidget(self.sphere_controls_widget)
        
        # Planar surface controls
        self.plane_controls_widget = QWidget()
        plane_layout = QVBoxLayout(self.plane_controls_widget)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        
        x_extent_layout = QHBoxLayout()
        x_extent_layout.addWidget(QLabel("X Extent:"))
        self.nf_x_extent_spin = QDoubleSpinBox()
        self.nf_x_extent_spin.setRange(0.01, 10.0)
        self.nf_x_extent_spin.setValue(0.5)
        self.nf_x_extent_spin.setSuffix(" m")
        self.nf_x_extent_spin.setDecimals(3)
        x_extent_layout.addWidget(self.nf_x_extent_spin)
        plane_layout.addLayout(x_extent_layout)
        
        y_extent_layout = QHBoxLayout()
        y_extent_layout.addWidget(QLabel("Y Extent:"))
        self.nf_y_extent_spin = QDoubleSpinBox()
        self.nf_y_extent_spin.setRange(0.01, 10.0)
        self.nf_y_extent_spin.setValue(0.5)
        self.nf_y_extent_spin.setSuffix(" m")
        self.nf_y_extent_spin.setDecimals(3)
        y_extent_layout.addWidget(self.nf_y_extent_spin)
        plane_layout.addLayout(y_extent_layout)
        
        z_dist_layout = QHBoxLayout()
        z_dist_layout.addWidget(QLabel("Z Distance:"))
        self.nf_z_distance_spin = QDoubleSpinBox()
        self.nf_z_distance_spin.setRange(0.001, 10.0)
        self.nf_z_distance_spin.setValue(0.1)
        self.nf_z_distance_spin.setSuffix(" m")
        self.nf_z_distance_spin.setDecimals(4)
        z_dist_layout.addWidget(self.nf_z_distance_spin)
        plane_layout.addLayout(z_dist_layout)
        
        x_pts_layout = QHBoxLayout()
        x_pts_layout.addWidget(QLabel("X Points:"))
        self.nf_x_points_spin = QSpinBox()
        self.nf_x_points_spin.setRange(10, 501)
        self.nf_x_points_spin.setValue(51)
        x_pts_layout.addWidget(self.nf_x_points_spin)
        plane_layout.addLayout(x_pts_layout)
        
        y_pts_layout = QHBoxLayout()
        y_pts_layout.addWidget(QLabel("Y Points:"))
        self.nf_y_points_spin = QSpinBox()
        self.nf_y_points_spin.setRange(10, 501)
        self.nf_y_points_spin.setValue(51)
        y_pts_layout.addWidget(self.nf_y_points_spin)
        plane_layout.addLayout(y_pts_layout)
        
        nf_group.addWidget(self.plane_controls_widget)
        
        # Initially hide plane controls
        self.plane_controls_widget.setVisible(False)
        
        # Calculate button
        self.calculate_nf_btn = QPushButton("Calculate Near Field")
        self.calculate_nf_btn.clicked.connect(self.on_calculate_nearfield)
        self.calculate_nf_btn.setEnabled(False)
        nf_group.addWidget(self.calculate_nf_btn)
        
        # Plot near field checkbox
        self.plot_nf_check = QCheckBox("Show Near Field in Plot")
        self.plot_nf_check.setChecked(True)
        self.plot_nf_check.setEnabled(False)
        nf_group.addWidget(self.plot_nf_check)
        
        # Results display
        self.nf_results = QTextEdit()
        self.nf_results.setReadOnly(True)
        self.nf_results.setMaximumHeight(80)
        self.nf_results.setPlaceholderText("Near field results will appear here...")
        nf_group.addWidget(self.nf_results)
        
        layout.addWidget(nf_group)
        
        # Add stretch
        layout.addStretch()
        
        # Set scroll widget
        scroll_area.setWidget(scroll_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        # Expand SWE by default
        swe_group.toggle_collapsed()
    
    def update_pattern(self, pattern):
        """Update controls with new pattern."""
        self.current_pattern = pattern
        self.swe_calculated = False
        self.nearfield_data = None
        
        # Update frequency combo for SWE
        self.swe_freq_combo.clear()
        for freq in pattern.frequencies:
            self.swe_freq_combo.addItem(f"{freq/1e6:.2f} MHz")
        
        # Enable SWE calculation
        self.calculate_swe_btn.setEnabled(True)
        
        # Disable near field until SWE is calculated
        self.calculate_nf_btn.setEnabled(False)
        self.plot_nf_check.setEnabled(False)
        
        # Clear results
        self.swe_results.clear()
        self.nf_results.clear()
    
    def on_swe_adaptive_toggled(self, checked):
        """Handle adaptive SWE checkbox toggle."""
        # Enable/disable radius input based on adaptive mode
        self.swe_radius_spin.setEnabled(not checked)
    
    def on_calculate_swe(self):
        """Handle calculate SWE button click."""
        if not self.current_pattern:
            return
        
        # Just emit the signal - let controls.py handle it with the worker thread
        self.calculate_swe_signal.emit()
    
    def on_surface_type_changed(self, surface_type):
        """Handle surface type change."""
        is_spherical = "Spherical" in surface_type
        self.sphere_controls_widget.setVisible(is_spherical)
        self.plane_controls_widget.setVisible(not is_spherical)
    
    def on_calculate_nearfield(self):
        """Handle calculate near field button click."""
        if not self.current_pattern or not self.swe_calculated:
            return
        
        self.calculate_nearfield_signal.emit()
    
    def display_swe_results(self, swe_data):
        """Display SWE calculation results."""
        self.swe_calculated = True
        
        result_text = f"SWE Coefficients Calculated:\n"
        result_text += f"Frequency: {swe_data['frequency']/1e9:.3f} GHz\n"
        result_text += f"Radius: {swe_data['radius']:.4f} m\n"
        result_text += f"Mode indices: M={swe_data['M']}, N={swe_data['N']}\n"
        result_text += f"Total power: {swe_data['mode_power']:.6e} W\n"
        
        if 'converged' in swe_data:
            result_text += f"Converged: {swe_data['converged']}\n"
            result_text += f"Iterations: {swe_data['iterations']}\n"
            result_text += f"Power retained: {swe_data.get('power_retained_fraction', 1.0):.2%}\n"
        
        self.swe_results.setText(result_text)
        
        # Enable near field calculation
        self.calculate_nf_btn.setEnabled(True)
        self.plot_nf_check.setEnabled(True)
    
    def display_nearfield_results(self, nf_data):
        """Display near field calculation results."""
        self.nearfield_data = nf_data
        
        surface_type = "spherical" if nf_data.get('is_spherical', True) else "planar"
        result_text = f"Near Field Calculated ({surface_type}):\n"
        
        if surface_type == "spherical":
            result_text += f"Radius: {nf_data['radius']:.4f} m\n"
            result_text += f"Theta points: {len(nf_data['theta'])}\n"
            result_text += f"Phi points: {len(nf_data['phi'])}\n"
        else:
            result_text += f"X extent: ±{nf_data['x_extent']:.3f} m\n"
            result_text += f"Y extent: ±{nf_data['y_extent']:.3f} m\n"
            result_text += f"Z distance: {nf_data['z_distance']:.4f} m\n"
            result_text += f"Grid: {len(nf_data['x'])} × {len(nf_data['y'])} points\n"
        
        self.nf_results.setText(result_text)
    
    # Getter methods for SWE parameters
    def get_swe_adaptive(self):
        """Get adaptive mode state."""
        return self.swe_adaptive_check.isChecked()
    
    def get_swe_radius(self):
        """Get SWE radius."""
        return self.swe_radius_spin.value()
    
    def get_swe_frequency(self):
        """Get selected frequency for SWE."""
        if self.current_pattern is None or self.swe_freq_combo.currentIndex() < 0:
            return None
        freq_index = self.swe_freq_combo.currentIndex()
        return self.current_pattern.frequencies[freq_index]
    
    # Getter methods for near field parameters
    def get_nf_surface_type(self):
        """Get near field surface type."""
        return "spherical" if "Spherical" in self.nf_surface_combo.currentText() else "planar"
    
    def get_nf_sphere_params(self):
        """Get spherical surface parameters."""
        return {
            'radius': self.nf_sphere_radius_spin.value(),
            'theta_points': self.nf_theta_points_spin.value(),
            'phi_points': self.nf_phi_points_spin.value()
        }
    
    def get_nf_plane_params(self):
        """Get planar surface parameters."""
        return {
            'x_extent': self.nf_x_extent_spin.value(),
            'y_extent': self.nf_y_extent_spin.value(),
            'z_distance': self.nf_z_distance_spin.value(),
            'x_points': self.nf_x_points_spin.value(),
            'y_points': self.nf_y_points_spin.value()
        }
    
    def get_plot_nearfield(self):
        """Get plot near field state."""
        return self.plot_nf_check.isChecked() and self.nearfield_data is not None