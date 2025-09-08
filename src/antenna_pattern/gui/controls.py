"""
Control panels for pattern visualization parameters.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QListWidget, QComboBox, QCheckBox, QLabel,
                            QAbstractItemView, QPushButton)
from PyQt6.QtCore import pyqtSignal


class ControlsWidget(QWidget):
    """Widget containing all plot control panels."""
    
    # Signal emitted when plot parameters change
    parameters_changed = pyqtSignal()
    
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
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
    
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
    
    def clear_all_controls(self):
        """Clear all control selections."""
        self.frequency_list.clear()
        self.phi_list.clear()
        self.polarization_combo.setItemText(0, "Current Pattern Polarization")
    
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