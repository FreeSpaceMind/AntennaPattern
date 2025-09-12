"""
Control panels for pattern visualization parameters.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QListWidget, QComboBox, QCheckBox, QLabel,
                            QAbstractItemView, QPushButton, QDoubleSpinBox)
from PyQt6.QtCore import pyqtSignal


class ControlsWidget(QWidget):
    """Widget containing all plot control panels."""
    
    # Signal emitted when plot parameters change
    parameters_changed = pyqtSignal()
    
    # Signals for pattern processing operations
    apply_phase_center_signal = pyqtSignal(float, float, float, float)  # x, y, z (meters), frequency
    apply_mars_signal = pyqtSignal(float)  # max_radial_extent
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the controls UI."""
        layout = QVBoxLayout()
        
        # Frequency selection
        freq_group = QGroupBox("Frequency Selection")
        freq_layout = QVBoxLayout()
        
        self.frequency_list = QListWidget()
        self.frequency_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.frequency_list.setMaximumHeight(120)
        self.frequency_list.itemSelectionChanged.connect(self.parameters_changed.emit)
        freq_layout.addWidget(self.frequency_list)
        
        # Frequency selection buttons
        freq_buttons = QHBoxLayout()
        self.freq_select_all = QPushButton("Select All")
        self.freq_select_all.clicked.connect(self.select_all_frequencies)
        self.freq_clear_all = QPushButton("Clear All")
        self.freq_clear_all.clicked.connect(self.clear_all_frequencies)
        freq_buttons.addWidget(self.freq_select_all)
        freq_buttons.addWidget(self.freq_clear_all)
        freq_layout.addLayout(freq_buttons)
        
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        # Phi angle selection  
        phi_group = QGroupBox("Phi Angle Selection")
        phi_layout = QVBoxLayout()
        
        self.phi_list = QListWidget()
        self.phi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.phi_list.setMaximumHeight(120)
        self.phi_list.itemSelectionChanged.connect(self.parameters_changed.emit)
        phi_layout.addWidget(self.phi_list)
        
        # Phi selection buttons
        phi_buttons = QHBoxLayout()
        self.phi_select_all = QPushButton("Select All")
        self.phi_select_all.clicked.connect(self.select_all_phi)
        self.phi_clear_all = QPushButton("Clear All") 
        self.phi_clear_all.clicked.connect(self.clear_all_phi)
        phi_buttons.addWidget(self.phi_select_all)
        phi_buttons.addWidget(self.phi_clear_all)
        phi_layout.addLayout(phi_buttons)
        
        phi_group.setLayout(phi_layout)
        layout.addWidget(phi_group)
        
        # Polarization selection
        pol_group = QGroupBox("Polarization")
        pol_layout = QVBoxLayout()
        
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems([
            "Current Pattern Polarization",
            "RHCP", "LHCP", "X (Ludwig-3)", "Y (Ludwig-3)", 
            "Theta", "Phi"
        ])
        self.polarization_combo.currentTextChanged.connect(self.parameters_changed.emit)
        pol_layout.addWidget(self.polarization_combo)
        
        pol_group.setLayout(pol_layout)
        layout.addWidget(pol_group)
        
        # Value type selection
        value_group = QGroupBox("Plot Type")
        value_layout = QVBoxLayout()
        
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Gain", "Phase", "Axial Ratio"])
        self.value_type_combo.currentTextChanged.connect(self.parameters_changed.emit)
        value_layout.addWidget(self.value_type_combo)
        
        # Cross-pol checkbox
        self.show_cross_pol = QCheckBox("Show Cross-Polarization")
        self.show_cross_pol.setChecked(True)
        self.show_cross_pol.toggled.connect(self.parameters_changed.emit)
        value_layout.addWidget(self.show_cross_pol)
        
        value_group.setLayout(value_layout)
        layout.addWidget(value_group)
        
        # Pattern Processing section
        processing_group = QGroupBox("Pattern Processing")
        processing_layout = QVBoxLayout()
        
        # Phase Center controls
        phase_center_label = QLabel("Phase Center Shifting:")
        phase_center_label.setStyleSheet("font-weight: bold;")
        processing_layout.addWidget(phase_center_label)
        
        # Theta angle input
        theta_layout = QHBoxLayout()
        theta_layout.addWidget(QLabel("Theta Angle:"))
        self.theta_angle_spin = QDoubleSpinBox()
        self.theta_angle_spin.setRange(1.0, 90.0)
        self.theta_angle_spin.setValue(30.0)
        self.theta_angle_spin.setSuffix("°")
        self.theta_angle_spin.setToolTip("Half-angle for phase center calculation (beamwidth/2)")
        theta_layout.addWidget(self.theta_angle_spin)
        processing_layout.addLayout(theta_layout)
        
        # Phase center frequency selection
        pc_freq_layout = QHBoxLayout()
        pc_freq_layout.addWidget(QLabel("PC Frequency:"))
        self.pc_freq_combo = QComboBox()
        self.pc_freq_combo.setToolTip("Frequency for phase center calculation")
        pc_freq_layout.addWidget(self.pc_freq_combo)
        processing_layout.addLayout(pc_freq_layout)
        
        # Find phase center button
        self.find_phase_center_btn = QPushButton("Find Phase Center")
        self.find_phase_center_btn.clicked.connect(self.on_find_phase_center)
        processing_layout.addWidget(self.find_phase_center_btn)
        
        # Manual phase center entry
        manual_pc_label = QLabel("Manual Phase Center (mm):")
        processing_layout.addWidget(manual_pc_label)
        
        pc_coords_layout = QHBoxLayout()
        pc_coords_layout.addWidget(QLabel("X:"))
        self.pc_x_spin = QDoubleSpinBox()
        self.pc_x_spin.setRange(-1000.0, 1000.0)
        self.pc_x_spin.setValue(0.0)
        self.pc_x_spin.setSuffix(" mm")
        self.pc_x_spin.setDecimals(2)
        pc_coords_layout.addWidget(self.pc_x_spin)
        
        pc_coords_layout.addWidget(QLabel("Y:"))
        self.pc_y_spin = QDoubleSpinBox()
        self.pc_y_spin.setRange(-1000.0, 1000.0)
        self.pc_y_spin.setValue(0.0)
        self.pc_y_spin.setSuffix(" mm")
        self.pc_y_spin.setDecimals(2)
        pc_coords_layout.addWidget(self.pc_y_spin)
        
        pc_coords_layout.addWidget(QLabel("Z:"))
        self.pc_z_spin = QDoubleSpinBox()
        self.pc_z_spin.setRange(-1000.0, 1000.0)
        self.pc_z_spin.setValue(0.0)
        self.pc_z_spin.setSuffix(" mm")
        self.pc_z_spin.setDecimals(2)
        pc_coords_layout.addWidget(self.pc_z_spin)
        
        processing_layout.addLayout(pc_coords_layout)
        
        # Apply phase center checkbox
        self.apply_phase_center_check = QCheckBox("Apply Phase Center Shift")
        self.apply_phase_center_check.toggled.connect(self.on_apply_phase_center_toggled)
        processing_layout.addWidget(self.apply_phase_center_check)
        
        # Phase center result display
        self.phase_center_result = QLabel("Phase center: Not calculated")
        self.phase_center_result.setStyleSheet("font-size: 9pt; color: #666;")
        processing_layout.addWidget(self.phase_center_result)
        
        # MARS controls
        mars_label = QLabel("MARS Algorithm:")
        mars_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        processing_layout.addWidget(mars_label)
        
        # Max radial extent input
        mre_layout = QHBoxLayout()
        mre_layout.addWidget(QLabel("Max Radial Extent:"))
        self.max_radial_extent_spin = QDoubleSpinBox()
        self.max_radial_extent_spin.setRange(0.001, 10.0)
        self.max_radial_extent_spin.setValue(0.5)
        self.max_radial_extent_spin.setSuffix(" m")
        self.max_radial_extent_spin.setDecimals(3)
        self.max_radial_extent_spin.setToolTip("Maximum radial extent of the antenna")
        mre_layout.addWidget(self.max_radial_extent_spin)
        processing_layout.addLayout(mre_layout)
        
        # MARS checkbox
        self.apply_mars_check = QCheckBox("Apply MARS")
        self.apply_mars_check.toggled.connect(self.on_apply_mars_toggled)
        processing_layout.addWidget(self.apply_mars_check)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Initially disable processing controls
        self.update_processing_controls_state()
    
    def update_pattern(self, pattern):
        """Update controls based on new pattern."""
        self.current_pattern = pattern
        
        if pattern is None:
            self.clear_all_controls()
            return
        
        # Update frequency list
        self.frequency_list.clear()
        frequencies = pattern.frequencies
        for freq in frequencies:
            freq_mhz = freq / 1e6
            item_text = f"{freq_mhz:.1f} MHz"
            self.frequency_list.addItem(item_text)
        
        # Select first frequency by default
        if self.frequency_list.count() > 0:
            self.frequency_list.setCurrentRow(0)
        
        # Update phi angle list
        self.phi_list.clear()
        phi_angles = pattern.phi_angles
        for phi in phi_angles:
            item_text = f"{phi:.1f}°"
            self.phi_list.addItem(item_text)
        
        # Select common phi angles (0 and 90) if they exist
        self.select_common_phi_angles()
        
        # Update polarization combo to show current pattern polarization
        current_pol = pattern.polarization.upper()
        combo_text = f"Current Pattern Polarization ({current_pol})"
        self.polarization_combo.setItemText(0, combo_text)
        self.polarization_combo.setCurrentIndex(0)
        
        # Update phase center frequency combo
        self.pc_freq_combo.clear()
        for freq in frequencies:
            freq_ghz = freq / 1e9
            item_text = f"{freq_ghz:.2f} GHz"
            self.pc_freq_combo.addItem(item_text)
        
        # Select middle frequency by default
        if self.pc_freq_combo.count() > 0:
            mid_idx = self.pc_freq_combo.count() // 2
            self.pc_freq_combo.setCurrentIndex(mid_idx)
        
        # Update processing controls state
        self.update_processing_controls_state()
        
        # Reset phase center result
        self.phase_center_result.setText("Phase center: Not calculated")
    
    def clear_all_controls(self):
        """Clear all control selections."""
        self.frequency_list.clear()
        self.phi_list.clear()
        self.pc_freq_combo.clear()
        self.polarization_combo.setItemText(0, "Current Pattern Polarization")
        self.phase_center_result.setText("Phase center: Not calculated")
        self.update_processing_controls_state()
    
    def update_processing_controls_state(self):
        """Enable/disable processing controls based on pattern availability."""
        has_pattern = self.current_pattern is not None
        self.find_phase_center_btn.setEnabled(has_pattern)
        self.apply_phase_center_check.setEnabled(has_pattern)
        self.apply_mars_check.setEnabled(has_pattern)
        self.theta_angle_spin.setEnabled(has_pattern)
        self.pc_freq_combo.setEnabled(has_pattern)
        self.max_radial_extent_spin.setEnabled(has_pattern)
        self.pc_x_spin.setEnabled(has_pattern)
        self.pc_y_spin.setEnabled(has_pattern)
        self.pc_z_spin.setEnabled(has_pattern)
    
    def select_all_frequencies(self):
        """Select all frequencies."""
        self.frequency_list.selectAll()
    
    def clear_all_frequencies(self):
        """Clear frequency selection."""
        self.frequency_list.clearSelection()
    
    def select_all_phi(self):
        """Select all phi angles."""
        self.phi_list.selectAll()
    
    def clear_all_phi(self):
        """Clear phi selection."""
        self.phi_list.clearSelection()
    
    def select_common_phi_angles(self):
        """Select common phi angles (0° and 90°) if they exist."""
        phi_angles = self.current_pattern.phi_angles if self.current_pattern else []
        
        for i, phi in enumerate(phi_angles):
            if abs(phi - 0.0) < 0.1 or abs(phi - 90.0) < 0.1:
                self.phi_list.item(i).setSelected(True)
    
    def get_selected_frequencies(self):
        """Get list of selected frequencies in Hz."""
        if not self.current_pattern:
            return []
            
        selected_indices = [item.row() for item in self.frequency_list.selectedIndexes()]
        frequencies = self.current_pattern.frequencies
        return [frequencies[i] for i in selected_indices]
    
    def get_selected_phi_angles(self):
        """Get list of selected phi angles in degrees."""
        if not self.current_pattern:
            return []
            
        selected_indices = [item.row() for item in self.phi_list.selectedIndexes()]
        phi_angles = self.current_pattern.phi_angles
        return [phi_angles[i] for i in selected_indices]
    
    def get_polarization(self):
        """Get selected polarization."""
        pol_text = self.polarization_combo.currentText()
        
        if pol_text.startswith("Current Pattern"):
            return None  # Use pattern's current polarization
        elif pol_text == "RHCP":
            return "rhcp"
        elif pol_text == "LHCP":
            return "lhcp"
        elif pol_text == "X (Ludwig-3)":
            return "x"
        elif pol_text == "Y (Ludwig-3)":
            return "y"
        elif pol_text == "Theta":
            return "theta"
        elif pol_text == "Phi":
            return "phi"
        else:
            return None
    
    def get_value_type(self):
        """Get selected value type."""
        value_text = self.value_type_combo.currentText()
        return value_text.lower().replace(" ", "_")  # "Axial Ratio" -> "axial_ratio"
    
    def get_show_cross_pol(self):
        """Get cross-polarization display setting."""
        # Don't show cross-pol for axial ratio
        if self.get_value_type() == "axial_ratio":
            return False
        return self.show_cross_pol.isChecked()
    
    def get_phase_center_frequency(self):
        """Get selected frequency for phase center calculation."""
        if not self.current_pattern or self.pc_freq_combo.currentIndex() < 0:
            return None
        freq_idx = self.pc_freq_combo.currentIndex()
        return self.current_pattern.frequencies[freq_idx]
    
    def get_manual_phase_center(self):
        """Get manually entered phase center coordinates in meters."""
        x_mm = self.pc_x_spin.value()
        y_mm = self.pc_y_spin.value()
        z_mm = self.pc_z_spin.value()
        return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]  # Convert mm to m
    
    def set_manual_phase_center(self, phase_center):
        """Set manual phase center coordinates from meters."""
        self.pc_x_spin.setValue(phase_center[0] * 1000.0)  # Convert m to mm
        self.pc_y_spin.setValue(phase_center[1] * 1000.0)
        self.pc_z_spin.setValue(phase_center[2] * 1000.0)
    
    def on_find_phase_center(self):
        """Handle find phase center button click."""
        if not self.current_pattern:
            return
            
        theta_angle = self.theta_angle_spin.value()
        frequency = self.get_phase_center_frequency()
        
        if frequency is None:
            return
        
        try:
            # Find phase center
            phase_center = self.current_pattern.find_phase_center(theta_angle, frequency)
            
            # Update manual entry fields
            self.set_manual_phase_center(phase_center)
            
            # Update display
            pc_text = f"Phase center: [{phase_center[0]*1000:.2f}, {phase_center[1]*1000:.2f}, {phase_center[2]*1000:.2f}] mm"
            self.phase_center_result.setText(pc_text)
            
        except Exception as e:
            self.phase_center_result.setText(f"Error: {str(e)}")
    
    def on_apply_phase_center_toggled(self, checked):
        """Handle apply phase center checkbox toggle."""
        if not self.current_pattern:
            return
            
        frequency = self.get_phase_center_frequency()
        if frequency is not None:
            phase_center = self.get_manual_phase_center()
            self.apply_phase_center_signal.emit(
                phase_center[0], phase_center[1], phase_center[2], frequency
            )

    def on_apply_mars_toggled(self, checked):
        """Handle apply MARS checkbox toggle.""" 
        if not self.current_pattern:
            return
            
        max_radial_extent = self.max_radial_extent_spin.value()
        self.apply_mars_signal.emit(max_radial_extent)