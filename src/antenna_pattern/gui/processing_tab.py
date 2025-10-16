"""
Processing tab - Controls for modifying pattern data.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QComboBox, QCheckBox, QPushButton, QDoubleSpinBox,
                            QScrollArea)
from PyQt6.QtCore import pyqtSignal, Qt

from .collapsible_group import CollapsibleGroupBox


class ProcessingTab(QWidget):
    """Tab containing pattern processing controls."""
    
    # Signals for processing operations
    apply_phase_center_signal = pyqtSignal(float, float, float, float)  # x, y, z, frequency
    apply_mars_signal = pyqtSignal(float)  # max_radial_extent
    polarization_changed = pyqtSignal()
    coordinate_format_changed = pyqtSignal(str)  # 'central' or 'sided'
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the processing tab UI."""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Create scrollable widget
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Polarization section
        pol_group = CollapsibleGroupBox("Polarization")
        
        pol_layout = QHBoxLayout()
        pol_layout.addWidget(QLabel("Polarization:"))
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["Theta", "Phi", "X (Ludwig-3)", 
                                         "Y (Ludwig-3)", "RHCP", "LHCP"])
        self.polarization_combo.currentTextChanged.connect(self.polarization_changed.emit)
        pol_layout.addWidget(self.polarization_combo)
        pol_group.addLayout(pol_layout)
        
        layout.addWidget(pol_group)

        # Coordinate Format section
        coord_group = CollapsibleGroupBox("Coordinate Format")

        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("Format:"))
        self.coord_format_combo = QComboBox()
        self.coord_format_combo.addItems(["Central", "Sided"])
        self.coord_format_combo.currentTextChanged.connect(self.on_coordinate_format_changed)
        coord_layout.addWidget(self.coord_format_combo)
        coord_group.addLayout(coord_layout)

        # Add description label
        desc_label = QLabel("Central: θ ±180°, φ 0-180°\nSided: θ 0-180°, φ 0-360°")
        desc_label.setStyleSheet("font-size: 9pt; color: #666;")
        coord_group.addWidget(desc_label)

        layout.addWidget(coord_group)
        
        # Phase center section (collapsible)
        pc_group = CollapsibleGroupBox("Phase Center")
        
        pc_group.addWidget(QLabel("Find phase center from far-field pattern:"))
        
        # Theta angle input
        theta_layout = QHBoxLayout()
        theta_layout.addWidget(QLabel("Theta Angle:"))
        self.theta_angle_spin = QDoubleSpinBox()
        self.theta_angle_spin.setRange(0.0, 90.0)
        self.theta_angle_spin.setValue(45.0)
        self.theta_angle_spin.setSuffix("°")
        theta_layout.addWidget(self.theta_angle_spin)
        pc_group.addLayout(theta_layout)
        
        # Frequency selection for phase center
        pc_freq_layout = QHBoxLayout()
        pc_freq_layout.addWidget(QLabel("Frequency:"))
        self.pc_freq_combo = QComboBox()
        pc_freq_layout.addWidget(self.pc_freq_combo)
        pc_group.addLayout(pc_freq_layout)
        
        # Find phase center button
        self.find_phase_center_btn = QPushButton("Find Phase Center")
        self.find_phase_center_btn.clicked.connect(self.on_find_phase_center)
        pc_group.addWidget(self.find_phase_center_btn)
        
        # Manual phase center coordinates
        pc_coords_layout = QHBoxLayout()
        pc_coords_layout.addWidget(QLabel("X:"))
        self.pc_x_spin = QDoubleSpinBox()
        self.pc_x_spin.setRange(-1000.0, 1000.0)
        self.pc_x_spin.setSuffix(" mm")
        self.pc_x_spin.setDecimals(2)
        self.pc_x_spin.setSingleStep(1.0)
        pc_coords_layout.addWidget(self.pc_x_spin)
        
        pc_coords_layout.addWidget(QLabel("Y:"))
        self.pc_y_spin = QDoubleSpinBox()
        self.pc_y_spin.setRange(-1000.0, 1000.0)
        self.pc_y_spin.setSuffix(" mm")
        self.pc_y_spin.setDecimals(2)
        self.pc_y_spin.setSingleStep(1.0)
        pc_coords_layout.addWidget(self.pc_y_spin)
        
        pc_coords_layout.addWidget(QLabel("Z:"))
        self.pc_z_spin = QDoubleSpinBox()
        self.pc_z_spin.setRange(-1000.0, 1000.0)
        self.pc_z_spin.setSuffix(" mm")
        self.pc_z_spin.setDecimals(2)
        self.pc_z_spin.setSingleStep(1.0)
        pc_coords_layout.addWidget(self.pc_z_spin)
        pc_group.addLayout(pc_coords_layout)
        
        # Apply phase center checkbox
        self.apply_phase_center_check = QCheckBox("Apply Phase Center Shift")
        self.apply_phase_center_check.toggled.connect(self.on_apply_phase_center_toggled)
        pc_group.addWidget(self.apply_phase_center_check)
        
        # Phase center result display
        self.phase_center_result = QLabel("Phase center: Not calculated")
        self.phase_center_result.setStyleSheet("font-size: 9pt; color: #666;")
        pc_group.addWidget(self.phase_center_result)
        
        layout.addWidget(pc_group)
        
        # MARS section (collapsible)
        mars_group = CollapsibleGroupBox("MARS Algorithm")
        
        mars_group.addWidget(QLabel("Minimum Antenna Radial Separation:"))
        
        # Max radial extent input
        mre_layout = QHBoxLayout()
        mre_layout.addWidget(QLabel("Max Radial Extent:"))
        self.max_radial_extent_spin = QDoubleSpinBox()
        self.max_radial_extent_spin.setRange(0.001, 10.0)
        self.max_radial_extent_spin.setValue(0.5)
        self.max_radial_extent_spin.setSuffix(" m")
        self.max_radial_extent_spin.setDecimals(3)
        mre_layout.addWidget(self.max_radial_extent_spin)
        mars_group.addLayout(mre_layout)
        
        # Apply MARS checkbox
        self.apply_mars_check = QCheckBox("Apply MARS")
        self.apply_mars_check.toggled.connect(self.on_apply_mars_toggled)
        mars_group.addWidget(self.apply_mars_check)
        
        layout.addWidget(mars_group)
        
        # Add stretch
        layout.addStretch()
        
        # Set scroll widget
        scroll_area.setWidget(scroll_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        # Expand polarization by default
        pol_group.toggle_collapsed()
        
        # Disable processing controls initially
        self.update_processing_controls_state()
    
    def update_pattern(self, pattern):
        """Update controls with new pattern."""
        self.current_pattern = pattern
        
        # Update frequency combo for phase center
        self.pc_freq_combo.clear()
        for freq in pattern.frequencies:
            self.pc_freq_combo.addItem(f"{freq/1e6:.2f} MHz")
        
        # Update polarization combo to match current pattern
        # Block signals to prevent triggering polarization_changed during initialization
        self.polarization_combo.blockSignals(True)
        pol_map = {
            'theta': 0,
            'phi': 1,
            'x': 2,
            'l3x': 2,
            'y': 3,
            'l3y': 3,
            'rhcp': 4,
            'rh': 4,
            'r': 4,
            'lhcp': 5,
            'lh': 5,
            'l': 5
        }
        idx = pol_map.get(pattern.polarization.lower(), 0)
        self.polarization_combo.setCurrentIndex(idx)
        self.polarization_combo.blockSignals(False)

        # coordinate format control
        from ..plotting import detect_coordinate_format
        current_format = detect_coordinate_format(pattern)
        format_idx = 0 if current_format == 'central' else 1
        self.coord_format_combo.blockSignals(True)
        self.coord_format_combo.setCurrentIndex(format_idx)
        self.coord_format_combo.blockSignals(False)
                

        
        # Enable processing controls
        self.update_processing_controls_state()
    
    def update_processing_controls_state(self):
        """Enable/disable processing controls based on pattern availability."""
        has_pattern = self.current_pattern is not None
        self.find_phase_center_btn.setEnabled(has_pattern)
        self.apply_phase_center_check.setEnabled(has_pattern)
        self.apply_mars_check.setEnabled(has_pattern)
    
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

    def on_coordinate_format_changed(self):
        """Handle coordinate format change."""
        format_map = {"Central": "central", "Sided": "sided"}
        format_type = format_map.get(self.coord_format_combo.currentText())
        if format_type:
            self.coordinate_format_changed.emit(format_type)
    
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
    
    def get_phase_center_frequency(self):
        """Get selected frequency for phase center calculation."""
        if self.current_pattern is None or self.pc_freq_combo.currentIndex() < 0:
            return None
        
        freq_index = self.pc_freq_combo.currentIndex()
        return self.current_pattern.frequencies[freq_index]
    
    def get_manual_phase_center(self):
        """Get manually entered phase center coordinates in meters."""
        x_mm = self.pc_x_spin.value()
        y_mm = self.pc_y_spin.value()
        z_mm = self.pc_z_spin.value()
        return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    
    def set_manual_phase_center(self, phase_center):
        """Set manual phase center coordinates from meters."""
        self.pc_x_spin.setValue(phase_center[0] * 1000.0)
        self.pc_y_spin.setValue(phase_center[1] * 1000.0)
        self.pc_z_spin.setValue(phase_center[2] * 1000.0)
    
    def get_polarization(self):
        """Get selected polarization type."""
        pol_map = {
            0: 'theta',
            1: 'phi',
            2: 'x',
            3: 'y',
            4: 'rhcp',
            5: 'lhcp'
        }
        return pol_map.get(self.polarization_combo.currentIndex())
    
    def reset_processing_state(self):
        """Reset checkboxes when loading new pattern."""
        self.apply_phase_center_check.setChecked(False)
        self.apply_mars_check.setChecked(False)