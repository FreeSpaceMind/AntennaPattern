"""
Control panels for pattern visualization parameters.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QListWidget, QComboBox, QCheckBox, QLabel,
                            QAbstractItemView, QPushButton, QDoubleSpinBox,
                            QScrollArea, QToolButton)
from PyQt6.QtCore import pyqtSignal, Qt

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
        """Setup the controls UI with scrollable collapsible sections."""
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create scrollable widget
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Frequency selection (collapsible)
        freq_group = CollapsibleGroupBox("Frequency Selection")
        
        self.frequency_list = QListWidget()
        self.frequency_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.frequency_list.setMaximumHeight(120)
        self.frequency_list.itemSelectionChanged.connect(self.parameters_changed.emit)
        freq_group.addWidget(self.frequency_list)
        
        # Frequency selection buttons
        freq_buttons = QHBoxLayout()
        self.freq_select_all = QPushButton("Select All")
        self.freq_select_all.clicked.connect(self.select_all_frequencies)
        self.freq_clear_all = QPushButton("Clear All")
        self.freq_clear_all.clicked.connect(self.clear_all_frequencies)
        freq_buttons.addWidget(self.freq_select_all)
        freq_buttons.addWidget(self.freq_clear_all)
        freq_group.addLayout(freq_buttons)
        
        scroll_layout.addWidget(freq_group)
        
        # Phi angle selection (collapsible)
        phi_group = CollapsibleGroupBox("Phi Angle Selection")
        
        self.phi_list = QListWidget()
        self.phi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.phi_list.setMaximumHeight(120)
        self.phi_list.itemSelectionChanged.connect(self.parameters_changed.emit)
        phi_group.addWidget(self.phi_list)
        
        # Phi selection buttons
        phi_buttons = QHBoxLayout()
        self.phi_select_all = QPushButton("Select All")
        self.phi_select_all.clicked.connect(self.select_all_phi)
        self.phi_clear_all = QPushButton("Clear All") 
        self.phi_clear_all.clicked.connect(self.clear_all_phi)
        phi_buttons.addWidget(self.phi_select_all)
        phi_buttons.addWidget(self.phi_clear_all)
        phi_group.addLayout(phi_buttons)
        
        scroll_layout.addWidget(phi_group)
        
        # Polarization selection (collapsible)
        pol_group = CollapsibleGroupBox("Polarization")
        
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems([
            "Current Pattern Polarization",
            "RHCP", "LHCP", "X (Ludwig-3)", "Y (Ludwig-3)", 
            "Theta", "Phi"
        ])
        self.polarization_combo.currentTextChanged.connect(self.parameters_changed.emit)
        pol_group.addWidget(self.polarization_combo)
        
        scroll_layout.addWidget(pol_group)
        
        # Value type selection (collapsible) - UPDATED
        value_group = CollapsibleGroupBox("Plot Type")
        
        # Plot format selection
        self.plot_format_combo = QComboBox()
        self.plot_format_combo.addItems(["1D Cut Plot", "2D Polar Plot"])
        self.plot_format_combo.currentTextChanged.connect(self.parameters_changed.emit)
        self.plot_format_combo.currentTextChanged.connect(self.on_plot_format_changed)
        value_group.addWidget(QLabel("Plot Format:"))
        value_group.addWidget(self.plot_format_combo)
        
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Gain", "Phase", "Axial Ratio"])
        self.value_type_combo.currentTextChanged.connect(self.parameters_changed.emit)
        value_group.addWidget(QLabel("Value Type:"))
        value_group.addWidget(self.value_type_combo)
        
        # REMOVE THIS ENTIRE SECTION:
        # # Component selection (for 2D plots)
        # self.component_combo = QComboBox()
        # self.component_combo.addItems(["Co-pol (e_co)", "Cross-pol (e_cx)", 
        #                               "E-theta (e_theta)", "E-phi (e_phi)"])
        # self.component_combo.currentTextChanged.connect(self.parameters_changed.emit)
        # value_group.addWidget(QLabel("Component:"))
        # value_group.addWidget(self.component_combo)
        
        # Cross-pol checkbox (only for 1D plots)
        self.show_cross_pol = QCheckBox("Show Cross-Polarization")
        self.show_cross_pol.setChecked(True)
        self.show_cross_pol.toggled.connect(self.parameters_changed.emit)
        value_group.addWidget(self.show_cross_pol)
        
        scroll_layout.addWidget(value_group)
                
        # Pattern Processing section (collapsible)
        processing_group = CollapsibleGroupBox("Pattern Processing")
        
        # Phase Center controls
        phase_center_label = QLabel("Phase Center Shifting:")
        phase_center_label.setStyleSheet("font-weight: bold;")
        processing_group.addWidget(phase_center_label)
        
        # Theta angle input
        theta_layout = QHBoxLayout()
        theta_layout.addWidget(QLabel("Theta Angle:"))
        self.theta_angle_spin = QDoubleSpinBox()
        self.theta_angle_spin.setRange(1.0, 90.0)
        self.theta_angle_spin.setValue(30.0)
        self.theta_angle_spin.setSuffix("°")
        self.theta_angle_spin.setToolTip("Angle for phase center optimization")
        theta_layout.addWidget(self.theta_angle_spin)
        processing_group.addLayout(theta_layout)
        
        # Phase center frequency selection
        pc_freq_layout = QHBoxLayout()
        pc_freq_layout.addWidget(QLabel("PC Frequency:"))
        self.pc_freq_combo = QComboBox()
        self.pc_freq_combo.setToolTip("Frequency for phase center calculation")
        pc_freq_layout.addWidget(self.pc_freq_combo)
        processing_group.addLayout(pc_freq_layout)
        
        # Find phase center button
        self.find_phase_center_btn = QPushButton("Find Phase Center")
        self.find_phase_center_btn.clicked.connect(self.on_find_phase_center)  # Use existing function name
        processing_group.addWidget(self.find_phase_center_btn)
        
        # Manual phase center coordinates
        pc_coords_layout = QHBoxLayout()
        pc_coords_layout.addWidget(QLabel("X:"))
        self.pc_x_spin = QDoubleSpinBox()
        self.pc_x_spin.setRange(-1.0, 1.0)
        self.pc_x_spin.setSuffix(" m")
        self.pc_x_spin.setDecimals(6)
        self.pc_x_spin.setSingleStep(0.001)
        pc_coords_layout.addWidget(self.pc_x_spin)
        
        pc_coords_layout.addWidget(QLabel("Y:"))
        self.pc_y_spin = QDoubleSpinBox()
        self.pc_y_spin.setRange(-1.0, 1.0)
        self.pc_y_spin.setSuffix(" m")
        self.pc_y_spin.setDecimals(6)
        self.pc_y_spin.setSingleStep(0.001)
        pc_coords_layout.addWidget(self.pc_y_spin)
        
        pc_coords_layout.addWidget(QLabel("Z:"))
        self.pc_z_spin = QDoubleSpinBox()
        self.pc_z_spin.setRange(-1.0, 1.0)
        self.pc_z_spin.setSuffix(" m")
        self.pc_z_spin.setDecimals(6)
        self.pc_z_spin.setSingleStep(0.001)
        pc_coords_layout.addWidget(self.pc_z_spin)
        processing_group.addLayout(pc_coords_layout)
        
        # Apply phase center checkbox
        self.apply_phase_center_check = QCheckBox("Apply Phase Center Shift")
        self.apply_phase_center_check.toggled.connect(self.on_apply_phase_center_toggled)
        processing_group.addWidget(self.apply_phase_center_check)
        
        # Phase center result display
        self.phase_center_result = QLabel("Phase center: Not calculated")
        self.phase_center_result.setStyleSheet("font-size: 9pt; color: #666;")
        processing_group.addWidget(self.phase_center_result)
        
        # MARS controls
        mars_label = QLabel("MARS Algorithm:")
        mars_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        processing_group.addWidget(mars_label)
        
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
        processing_group.addLayout(mre_layout)
        
        # MARS checkbox
        self.apply_mars_check = QCheckBox("Apply MARS")
        self.apply_mars_check.toggled.connect(self.on_apply_mars_toggled)
        processing_group.addWidget(self.apply_mars_check)
        
        scroll_layout.addWidget(processing_group)

        # Statistics section (collapsible)
        stats_group = CollapsibleGroupBox("Plot Statistics")

        # Enable statistics checkbox
        self.enable_statistics = QCheckBox("Enable Statistics Plot")
        self.enable_statistics.setChecked(False)
        self.enable_statistics.toggled.connect(self.parameters_changed.emit)
        stats_group.addWidget(self.enable_statistics)

        # Show range checkbox
        self.show_range = QCheckBox("Show Min/Max Range")
        self.show_range.setChecked(True)
        self.show_range.toggled.connect(self.parameters_changed.emit)
        stats_group.addWidget(self.show_range)

        # Statistic type dropdown
        statistic_layout = QHBoxLayout()
        statistic_layout.addWidget(QLabel("Statistic:"))
        self.statistic_combo = QComboBox()
        self.statistic_combo.addItems(["mean", "median", "rms", "percentile", "std"])
        self.statistic_combo.setCurrentText("mean")
        self.statistic_combo.currentTextChanged.connect(self.parameters_changed.emit)
        self.statistic_combo.currentTextChanged.connect(self.on_statistic_changed)
        statistic_layout.addWidget(self.statistic_combo)
        stats_group.addLayout(statistic_layout)

        # Percentile range inputs
        percentile_layout = QHBoxLayout()
        percentile_layout.addWidget(QLabel("Percentile Range:"))
        self.percentile_lower_spin = QDoubleSpinBox()
        self.percentile_lower_spin.setRange(0.0, 100.0)
        self.percentile_lower_spin.setValue(25.0)
        self.percentile_lower_spin.setSuffix("%")
        self.percentile_lower_spin.valueChanged.connect(self.parameters_changed.emit)
        percentile_layout.addWidget(self.percentile_lower_spin)

        percentile_layout.addWidget(QLabel("to"))

        self.percentile_upper_spin = QDoubleSpinBox()
        self.percentile_upper_spin.setRange(0.0, 100.0)
        self.percentile_upper_spin.setValue(75.0)
        self.percentile_upper_spin.setSuffix("%")
        self.percentile_upper_spin.valueChanged.connect(self.parameters_changed.emit)
        percentile_layout.addWidget(self.percentile_upper_spin)
        stats_group.addLayout(percentile_layout)

        # Initially hide percentile controls
        self.percentile_lower_spin.setVisible(False)
        self.percentile_upper_spin.setVisible(False)
        percentile_layout.itemAt(0).widget().setVisible(False)  # "Percentile Range:" label
        percentile_layout.itemAt(2).widget().setVisible(False)  # "to" label

        scroll_layout.addWidget(stats_group)
                
        # Add stretch to push everything to top
        scroll_layout.addStretch()
        
        # Set scroll widget
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        self.setLayout(main_layout)
        
        # Initially disable processing controls
        self.update_processing_controls_state()
        
        # Expand frequently used sections by default
        freq_group.toggle_collapsed()  # Expand frequency selection
        value_group.toggle_collapsed()  # Expand plot type selection
    
    def update_pattern(self, pattern):
        """Update controls based on new pattern."""
        self.current_pattern = pattern
        
        if pattern is None:
            self.clear_all_controls()
            return
        
        # Clear existing items
        self.frequency_list.clear()
        self.phi_list.clear()
        self.pc_freq_combo.clear()  # Clear phase center frequency combo

        # Update frequency list and phase center frequency combo
        for freq in pattern.frequencies:
            freq_mhz = freq / 1e6
            freq_item = f"{freq_mhz:.1f} MHz"
            self.frequency_list.addItem(freq_item)
            self.pc_freq_combo.addItem(freq_item)  # Add to phase center combo

        # Update phi list
        for phi in pattern.phi_angles:
            self.phi_list.addItem(f"{phi:.1f}°")
        
        # Select first frequency by default in both lists
        if self.frequency_list.count() > 0:
            self.frequency_list.item(0).setSelected(True)
            self.pc_freq_combo.setCurrentIndex(0)
        
        # Select first phi by default
        if self.phi_list.count() > 0:
            self.phi_list.item(0).setSelected(True)
        
        # Update polarization combo to show current pattern polarization
        current_pol = pattern.polarization.upper()
        combo_text = f"Current Pattern Polarization ({current_pol})"
        self.polarization_combo.setItemText(0, combo_text)
        self.polarization_combo.setCurrentIndex(0)
        
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
        """Get the selected phase center frequency in Hz."""
        if self.current_pattern is None or self.pc_freq_combo.currentIndex() < 0:
            return None
        
        freq_index = self.pc_freq_combo.currentIndex()
        return self.current_pattern.frequencies[freq_index]
    
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

    def get_plot_format(self):
        """Get selected plot format."""
        format_text = self.plot_format_combo.currentText()
        return "2d_polar" if "2D Polar" in format_text else "1d_cut"

    def on_plot_format_changed(self):
        """Handle plot format change."""
        self.update_controls_for_plot_format()
        self.parameters_changed.emit()

    def update_controls_for_plot_format(self):
        """Update control enabling/disabling based on plot format dropdown."""
        is_2d = self.get_plot_format() == "2d_polar"
        
        print(f"DEBUG: ControlsWidget updating for format: {self.get_plot_format()}, is_2d: {is_2d}")
        
        # For 2D plots, cross-pol doesn't make sense, so disable it
        self.show_cross_pol.setEnabled(not is_2d)
        if is_2d:
            self.show_cross_pol.setChecked(False)
            
        # For 2D plots, limit frequency selection to single frequency
        if is_2d:
            # If multiple frequencies selected, keep only the first one
            selected_items = self.frequency_list.selectedItems()
            if len(selected_items) > 1:
                self.frequency_list.clearSelection()
                selected_items[0].setSelected(True) 
                
        # Statistics only available for 1D plots
        is_1d = (self.get_plot_format() == '1d_cut')
        if hasattr(self, 'enable_statistics'):
            # Find and show/hide the entire statistics group
            # Statistics controls should only be visible for 1D plots
            stats_widgets = [
                self.enable_statistics, self.show_range, 
                self.statistic_combo, self.percentile_lower_spin, 
                self.percentile_upper_spin
            ]
            
            for widget in stats_widgets:
                widget.setVisible(is_1d)

    def get_plot_format(self):
        """Get selected plot format."""
        format_text = self.plot_format_combo.currentText()
        return "2d_polar" if "2D Polar" in format_text else "1d_cut"

    def get_component(self):
        """Get field component based on selected polarization."""
        polarization = self.get_polarization()
        
        # Map polarization to appropriate component
        if polarization is None or polarization in ['Current Pattern Polarization']:
            # Use co-pol for current pattern polarization
            return 'e_co'
        elif polarization.lower() in ['rhcp', 'rh', 'r', 'lhcp', 'lh', 'l']:
            # Circular polarizations - use co-pol
            return 'e_co'
        elif polarization.lower() in ['x', 'l3x', 'ludwig-3']:
            # X polarization - use theta component (typical mapping)
            return 'e_theta'
        elif polarization.lower() in ['y', 'l3y']:
            # Y polarization - use phi component (typical mapping)  
            return 'e_phi'
        elif polarization.lower() == 'theta':
            # Theta polarization
            return 'e_theta'
        elif polarization.lower() == 'phi':
            # Phi polarization
            return 'e_phi'
        else:
            # Default to co-pol
            return 'e_co'
        
    def get_polarization(self):
        """Get selected polarization from combo box."""
        pol_text = self.polarization_combo.currentText()
        
        # Map display text to internal polarization names
        pol_map = {
            "Current Pattern Polarization": None,
            "RHCP": "rhcp",
            "LHCP": "lhcp", 
            "X (Ludwig-3)": "x",
            "Y (Ludwig-3)": "y",
            "Theta": "theta",
            "Phi": "phi"
        }
        
        return pol_map.get(pol_text, None)
    
    def on_statistic_changed(self):
        """Handle statistic type change to show/hide percentile controls."""
        is_percentile = self.statistic_combo.currentText() == "percentile"
        
        # Show/hide percentile controls
        self.percentile_lower_spin.setVisible(is_percentile)
        self.percentile_upper_spin.setVisible(is_percentile)
        
        # Find the percentile layout and show/hide labels
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if hasattr(item, 'layout'):
                layout = item.layout()
                if layout and layout.count() >= 4:
                    # Check if this looks like our percentile layout
                    label_item = layout.itemAt(0)
                    if (label_item and label_item.widget() and 
                        hasattr(label_item.widget(), 'text') and 
                        "Percentile Range:" in label_item.widget().text()):
                        
                        label_item.widget().setVisible(is_percentile)
                        layout.itemAt(2).widget().setVisible(is_percentile)  # "to" label
                        break

    def get_statistics_enabled(self):
        """Get whether statistics plotting is enabled."""
        return self.enable_statistics.isChecked()

    def get_show_range(self):
        """Get show range setting."""
        return self.show_range.isChecked()

    def get_statistic_type(self):
        """Get selected statistic type."""
        return self.statistic_combo.currentText()

    def get_percentile_range(self):
        """Get percentile range as tuple."""
        return (self.percentile_lower_spin.value(), self.percentile_upper_spin.value())