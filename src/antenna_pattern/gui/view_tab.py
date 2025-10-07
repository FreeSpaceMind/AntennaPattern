"""
View tab - Controls for visualizing pattern data.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QListWidget, QComboBox, QCheckBox, QLabel,
                            QAbstractItemView, QPushButton, QDoubleSpinBox,
                            QScrollArea)
from PyQt6.QtCore import pyqtSignal, Qt

from .collapsible_group import CollapsibleGroupBox


class ViewTab(QWidget):
    """Tab containing visualization controls."""
    
    # Signal emitted when any parameter changes
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the view tab UI."""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Create scrollable widget
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        
        # Frequency selection (collapsible)
        freq_group = CollapsibleGroupBox("Frequency Selection")
        
        self.frequency_list = QListWidget()
        self.frequency_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.frequency_list.setMaximumHeight(120)
        self.frequency_list.itemSelectionChanged.connect(self.parameters_changed.emit)
        freq_group.addWidget(self.frequency_list)
        
        freq_buttons = QHBoxLayout()
        self.freq_select_all = QPushButton("Select All")
        self.freq_select_all.clicked.connect(self.select_all_frequencies)
        self.freq_clear_all = QPushButton("Clear All")
        self.freq_clear_all.clicked.connect(self.clear_all_frequencies)
        freq_buttons.addWidget(self.freq_select_all)
        freq_buttons.addWidget(self.freq_clear_all)
        freq_group.addLayout(freq_buttons)
        
        layout.addWidget(freq_group)
        
        # Phi angle selection (collapsible)
        phi_group = CollapsibleGroupBox("Phi Angle Selection")
        
        self.phi_list = QListWidget()
        self.phi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.phi_list.setMaximumHeight(120)
        self.phi_list.itemSelectionChanged.connect(self.parameters_changed.emit)
        phi_group.addWidget(self.phi_list)
        
        phi_buttons = QHBoxLayout()
        self.phi_select_all = QPushButton("Select All")
        self.phi_select_all.clicked.connect(self.select_all_phi)
        self.phi_clear_all = QPushButton("Clear All")
        self.phi_clear_all.clicked.connect(self.clear_all_phi)
        phi_buttons.addWidget(self.phi_select_all)
        phi_buttons.addWidget(self.phi_clear_all)
        phi_group.addLayout(phi_buttons)
        
        layout.addWidget(phi_group)
        
        # Plot settings (collapsible)
        plot_group = CollapsibleGroupBox("Plot Settings")
        
        # Plot format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.plot_format_combo = QComboBox()
        self.plot_format_combo.addItems(["1D Cut", "2D Polar", "Near Field"])
        self.plot_format_combo.currentTextChanged.connect(self.on_plot_format_changed)
        format_layout.addWidget(self.plot_format_combo)
        plot_group.addLayout(format_layout)
        
        # Value type
        value_layout = QHBoxLayout()
        value_layout.addWidget(QLabel("Value:"))
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Gain", "Phase", "Axial Ratio"])
        self.value_type_combo.currentTextChanged.connect(self.parameters_changed.emit)
        value_layout.addWidget(self.value_type_combo)
        plot_group.addLayout(value_layout)
        
        # Component selection
        component_layout = QHBoxLayout()
        component_layout.addWidget(QLabel("Component:"))
        self.component_combo = QComboBox()
        self.component_combo.addItems(["Co-pol", "Cross-pol", "E-theta", "E-phi"])
        self.component_combo.currentTextChanged.connect(self.parameters_changed.emit)
        component_layout.addWidget(self.component_combo)
        plot_group.addLayout(component_layout)
        
        # Show cross-pol checkbox
        self.show_cross_pol = QCheckBox("Show Cross-Polarization")
        self.show_cross_pol.setChecked(False)
        self.show_cross_pol.toggled.connect(self.parameters_changed.emit)
        plot_group.addWidget(self.show_cross_pol)
        
        layout.addWidget(plot_group)
        
        # Statistics (collapsible)
        stats_group = CollapsibleGroupBox("Plot Statistics")
        
        self.enable_statistics = QCheckBox("Enable Statistics Plot")
        self.enable_statistics.setChecked(False)
        self.enable_statistics.toggled.connect(self.parameters_changed.emit)
        stats_group.addWidget(self.enable_statistics)
        
        self.show_range = QCheckBox("Show Min/Max Range")
        self.show_range.setChecked(True)
        self.show_range.toggled.connect(self.parameters_changed.emit)
        stats_group.addWidget(self.show_range)
        
        statistic_layout = QHBoxLayout()
        statistic_layout.addWidget(QLabel("Statistic:"))
        self.statistic_combo = QComboBox()
        self.statistic_combo.addItems(["mean", "median", "rms", "percentile", "std"])
        self.statistic_combo.currentTextChanged.connect(self.parameters_changed.emit)
        self.statistic_combo.currentTextChanged.connect(self.on_statistic_changed)
        statistic_layout.addWidget(self.statistic_combo)
        stats_group.addLayout(statistic_layout)
        
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
        
        # Hide percentile controls initially
        self.percentile_lower_spin.setVisible(False)
        self.percentile_upper_spin.setVisible(False)
        percentile_layout.itemAt(0).widget().setVisible(False)
        percentile_layout.itemAt(2).widget().setVisible(False)
        
        layout.addWidget(stats_group)
        
        # Add stretch
        layout.addStretch()
        
        # Set scroll widget
        scroll_area.setWidget(scroll_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        # Expand frequently used sections
        freq_group.toggle_collapsed()
        plot_group.toggle_collapsed()
    
    def update_pattern(self, pattern):
        """Update controls with new pattern."""
        self.current_pattern = pattern
        
        # Update frequency list
        self.frequency_list.clear()
        for freq in pattern.frequencies:
            self.frequency_list.addItem(f"{freq/1e6:.2f} MHz")
        self.frequency_list.setCurrentRow(0)
        
        # Update phi list
        self.phi_list.clear()
        for phi in pattern.phi_angles:
            self.phi_list.addItem(f"{phi:.1f}°")
        self.phi_list.setCurrentRow(0)
    
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
    
    def on_plot_format_changed(self):
        """Handle plot format change."""
        self.parameters_changed.emit()
    
    def on_statistic_changed(self, statistic):
        """Handle statistic type change."""
        # Show/hide percentile controls
        is_percentile = statistic == "percentile"
        self.percentile_lower_spin.setVisible(is_percentile)
        self.percentile_upper_spin.setVisible(is_percentile)
        # Get the layout items
        percentile_layout = self.percentile_lower_spin.parent().layout()
        if percentile_layout:
            for i in range(percentile_layout.count()):
                item = percentile_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    item.widget().setVisible(is_percentile)
    
    # Getter methods
    def get_selected_frequencies(self):
        """Get list of selected frequencies."""
        if self.current_pattern is None:
            return []
        selected_items = self.frequency_list.selectedItems()
        if not selected_items:
            return [self.current_pattern.frequencies[0]]
        indices = [self.frequency_list.row(item) for item in selected_items]
        return [self.current_pattern.frequencies[i] for i in indices]
    
    def get_selected_phi_angles(self):
        """Get list of selected phi angles."""
        if self.current_pattern is None:
            return []
        selected_items = self.phi_list.selectedItems()
        if not selected_items:
            return [self.current_pattern.phi_angles[0]]
        indices = [self.phi_list.row(item) for item in selected_items]
        return [self.current_pattern.phi_angles[i] for i in indices]
    
    def get_plot_format(self):
        """Get selected plot format."""
        format_text = self.plot_format_combo.currentText()
        if "2D Polar" in format_text:
            return "2d_polar"
        elif "Near Field" in format_text:
            return "near_field"
        else:
            return "1d_cut"
    
    def get_value_type(self):
        """Get selected value type."""
        return self.value_type_combo.currentText().lower().replace(" ", "_")
    
    def get_component(self):
        """Get selected component."""
        component_map = {
            "Co-pol": "e_co",
            "Cross-pol": "e_cx",
            "E-theta": "e_theta",
            "E-phi": "e_phi"
        }
        return component_map[self.component_combo.currentText()]
    
    def get_show_cross_pol(self):
        """Get show cross-pol state."""
        return self.show_cross_pol.isChecked()
    
    def get_statistics_enabled(self):
        """Get statistics enabled state."""
        return self.enable_statistics.isChecked()
    
    def get_show_range(self):
        """Get show range state."""
        return self.show_range.isChecked()
    
    def get_statistic_type(self):
        """Get selected statistic type."""
        return self.statistic_combo.currentText()
    
    def get_percentile_range(self):
        """Get percentile range."""
        return (self.percentile_lower_spin.value(), self.percentile_upper_spin.value())