"""
Main window for the Antenna Pattern GUI.
"""

import os
from pathlib import Path

from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                            QMenuBar, QFileDialog, QMessageBox, QSplitter,
                            QStatusBar, QLabel, QDialog, QFormLayout, 
                            QDoubleSpinBox, QDialogButtonBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from .plot_widget import PlotWidget
from .controls import ControlsWidget
from ..ant_io import read_cut, read_ffd, load_pattern_npz, save_pattern_npz
from ..pattern import AntennaPattern


class MainWindow(QMainWindow):
    """Main window for the Antenna Pattern GUI application."""
    
    def __init__(self):
        super().__init__()
        self.current_pattern = None
        self.current_file_path = None
        
        self.setup_ui()
        self.create_menus()
        self.create_status_bar()
        
        # Timer for delayed plot updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)
        
    def setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("Antenna Pattern Analyzer")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create controls widget (left side)
        self.controls = ControlsWidget()
        self.controls.parameters_changed.connect(self.on_parameters_changed)
        # Connect new processing signals
        self.controls.apply_phase_center_signal.connect(self.on_apply_phase_center)
        self.controls.apply_mars_signal.connect(self.on_apply_mars)
        self.controls.setMaximumWidth(350)  # Increased width for new controls
        self.controls.setMinimumWidth(300)
        
        # Create plot widget (right side)
        self.plot_widget = PlotWidget()
        
        # Add widgets to splitter
        splitter.addWidget(self.controls)
        splitter.addWidget(self.plot_widget)
        
        # Set splitter proportions (controls smaller, plot larger)
        splitter.setSizes([300, 1100])
        
        # Add splitter to main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
    
    def create_menus(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Import actions
        import_cut_action = QAction("Import CUT File...", self)
        import_cut_action.triggered.connect(self.import_cut_file)
        file_menu.addAction(import_cut_action)
        
        import_ffd_action = QAction("Import FFD File...", self)
        import_ffd_action.triggered.connect(self.import_ffd_file)
        file_menu.addAction(import_ffd_action)
        
        import_npz_action = QAction("Import NPZ File...", self)
        import_npz_action.triggered.connect(self.import_npz_file)
        file_menu.addAction(import_npz_action)
        
        file_menu.addSeparator()
        
        # Export actions
        export_npz_action = QAction("Export NPZ File...", self)
        export_npz_action.triggered.connect(self.export_npz_file)
        file_menu.addAction(export_npz_action)
        
        export_cut_action = QAction("Export CUT File...", self)
        export_cut_action.triggered.connect(self.export_cut_file)
        file_menu.addAction(export_cut_action)
        
        # Store export actions to enable/disable them
        self.export_actions = [export_npz_action, export_cut_action]
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """Create the status bar."""
        self.statusBar().showMessage("Ready")
        
        # Add permanent widgets to status bar
        self.pattern_info_label = QLabel("No pattern loaded")
        self.file_path_label = QLabel("")
        
        self.statusBar().addWidget(self.pattern_info_label)
        self.statusBar().addPermanentWidget(self.file_path_label)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Antenna Pattern Analyzer",
                         "Antenna Pattern Analyzer\n\n"
                         "A tool for visualizing and analyzing antenna radiation patterns.\n\n"
                         "Supports CUT, FFD, and NPZ file formats.")
    
    def update_export_actions(self):
        """Enable/disable export actions based on pattern availability."""
        has_pattern = self.current_pattern is not None
        for action in self.export_actions:
            action.setEnabled(has_pattern)
    
    def update_status_bar(self):
        """Update status bar with current pattern info."""
        if self.current_pattern is None:
            self.pattern_info_label.setText("No pattern loaded")
            self.file_path_label.setText("")
        else:
            freq_range = f"{self.current_pattern.frequencies[0]/1e6:.1f}-{self.current_pattern.frequencies[-1]/1e6:.1f} MHz"
            theta_range = f"{self.current_pattern.theta_angles[0]:.1f}°-{self.current_pattern.theta_angles[-1]:.1f}°"
            phi_range = f"{self.current_pattern.phi_angles[0]:.1f}°-{self.current_pattern.phi_angles[-1]:.1f}°"
            
            info_text = (f"Frequencies: {freq_range} | "
                        f"Theta: {theta_range} | "
                        f"Phi: {phi_range} | "
                        f"Polarization: {self.current_pattern.polarization.upper()}")
            
            self.pattern_info_label.setText(info_text)
            
            if self.current_file_path:
                self.file_path_label.setText(str(self.current_file_path))
    
    def on_parameters_changed(self):
        """Handle parameter changes with delayed update."""
        # Stop any existing timer and restart with delay to avoid excessive updates
        self.update_timer.stop()
        self.update_timer.start(100)  # 100ms delay
    
    def update_plot(self):
        """Update the plot with current parameters."""
        if self.current_pattern is None:
            self.plot_widget.clear_plot()
            return
        
        # Get current parameters from controls
        frequencies = self.controls.get_selected_frequencies()
        phi_angles = self.controls.get_selected_phi_angles()
        polarization = self.controls.get_polarization()
        value_type = self.controls.get_value_type()
        show_cross_pol = self.controls.get_show_cross_pol()
        
        # Convert to format expected by plot function
        freq_list = frequencies if len(frequencies) > 1 else (frequencies[0] if frequencies else None)
        phi_list = phi_angles if len(phi_angles) > 1 else (phi_angles[0] if phi_angles else None)
        
        # Get pattern with desired polarization
        plot_pattern = self.current_pattern
        if polarization is not None:
            # Create a copy with different polarization
            try:
                plot_pattern = AntennaPattern(
                    theta=self.current_pattern.theta_angles,
                    phi=self.current_pattern.phi_angles,
                    frequency=self.current_pattern.frequencies,
                    e_theta=self.current_pattern.data.e_theta.values.copy(),
                    e_phi=self.current_pattern.data.e_phi.values.copy(),
                    polarization=polarization
                )
            except Exception as e:
                self.show_error(f"Error changing polarization: {str(e)}")
                return
        
        # Update plot
        self.plot_widget.update_plot(
            pattern=plot_pattern,
            frequencies=freq_list,
            phi_angles=phi_list,
            value_type=value_type,
            show_cross_pol=show_cross_pol
        )
    
    def load_pattern(self, pattern, file_path=None):
        """Load a new pattern and update the GUI."""
        self.current_pattern = pattern
        self.current_file_path = file_path
        
        # Update controls with new pattern
        self.controls.update_pattern(pattern)
        
        # Update status bar
        self.update_status_bar()
        
        # Update export actions
        self.update_export_actions()
        
        # Update plot
        self.update_plot()
    
    def on_apply_phase_center(self, x, y, z, frequency):
        """Handle phase center application."""
        if not self.current_pattern:
            return
        
        try:
            # Apply phase center shift using manual coordinates
            self.statusBar().showMessage("Applying phase center shift...")
            
            # Create translation vector from coordinates (already in meters)
            translation = [x, y, z]
            
            # Apply translation directly
            self.current_pattern.translate(translation, normalize=True)
            
            # Update the plot
            self.update_plot()
            
            # Update status
            translation_mm = [x*1000, y*1000, z*1000]  # Convert to mm for display
            self.statusBar().showMessage(
                f"Phase center shift applied. Translation: "
                f"[{translation_mm[0]:.2f}, {translation_mm[1]:.2f}, {translation_mm[2]:.2f}] mm"
            )
            
            # Update the phase center result display
            self.controls.phase_center_result.setText(
                f"Applied shift: [{translation_mm[0]:.2f}, {translation_mm[1]:.2f}, {translation_mm[2]:.2f}] mm"
            )
            
            # Uncheck the checkbox after applying
            self.controls.apply_phase_center_check.setChecked(False)
            
        except Exception as e:
            self.show_error(f"Error applying phase center shift: {str(e)}")
            self.statusBar().showMessage("Ready")
            # Uncheck the checkbox on error
            self.controls.apply_phase_center_check.setChecked(False)
    
    def on_apply_mars(self, max_radial_extent):
        """Handle MARS application."""
        if not self.current_pattern:
            return
        
        try:
            # Apply MARS algorithm
            self.statusBar().showMessage("Applying MARS algorithm...")
            
            # Apply MARS in-place
            self.current_pattern.apply_mars(max_radial_extent)
            
            # Update the plot
            self.update_plot()
            
            # Update status
            self.statusBar().showMessage(
                f"MARS algorithm applied with max radial extent: {max_radial_extent:.3f} m"
            )
            
            # Uncheck the checkbox after applying
            self.controls.apply_mars_check.setChecked(False)
            
        except Exception as e:
            self.show_error(f"Error applying MARS: {str(e)}")
            self.statusBar().showMessage("Ready")
            # Uncheck the checkbox on error
            self.controls.apply_mars_check.setChecked(False)
    
    def import_cut_file(self):
        """Import a CUT file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import CUT File", "", "CUT Files (*.cut);;All Files (*)"
        )
        
        if file_path:
            # Get frequency range from user
            dialog = FrequencyRangeDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                freq_start, freq_end = dialog.get_frequency_range()
                
                try:
                    self.statusBar().showMessage("Loading CUT file...")
                    pattern = read_cut(file_path, freq_start, freq_end)
                    self.load_pattern(pattern, file_path)
                    self.statusBar().showMessage("CUT file loaded successfully")
                except Exception as e:
                    self.show_error(f"Error loading CUT file: {str(e)}")
                    self.statusBar().showMessage("Ready")
    
    def import_ffd_file(self):
        """Import an FFD file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import FFD File", "", "FFD Files (*.ffd);;All Files (*)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage("Loading FFD file...")
                pattern = read_ffd(file_path)
                self.load_pattern(pattern, file_path)
                self.statusBar().showMessage("FFD file loaded successfully")
            except Exception as e:
                self.show_error(f"Error loading FFD file: {str(e)}")
                self.statusBar().showMessage("Ready")
    
    def import_npz_file(self):
        """Import an NPZ file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import NPZ File", "", "NPZ Files (*.npz);;All Files (*)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage("Loading NPZ file...")
                pattern, metadata = load_pattern_npz(file_path)
                self.load_pattern(pattern, file_path)
                self.statusBar().showMessage("NPZ file loaded successfully")
            except Exception as e:
                self.show_error(f"Error loading NPZ file: {str(e)}")
                self.statusBar().showMessage("Ready")
    
    def export_npz_file(self):
        """Export pattern to NPZ file."""
        if not self.current_pattern:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export NPZ File", "", "NPZ Files (*.npz);;All Files (*)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage("Saving NPZ file...")
                metadata = {"exported_from": "Antenna Pattern GUI"}
                save_pattern_npz(self.current_pattern, file_path, metadata)
                self.statusBar().showMessage("NPZ file saved successfully")
            except Exception as e:
                self.show_error(f"Error saving NPZ file: {str(e)}")
                self.statusBar().showMessage("Ready")
    
    def export_cut_file(self):
        """Export pattern to CUT file."""
        if not self.current_pattern:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CUT File", "", "CUT Files (*.cut);;All Files (*)"
        )
        
        if file_path:
            try:
                self.statusBar().showMessage("Saving CUT file...")
                self.current_pattern.write_cut(file_path)
                self.statusBar().showMessage("CUT file saved successfully")
            except Exception as e:
                self.show_error(f"Error saving CUT file: {str(e)}")
                self.statusBar().showMessage("Ready")
    
    def show_error(self, message):
        """Show an error message."""
        QMessageBox.critical(self, "Error", message)
    
    def show_info(self, message):
        """Show an info message."""
        QMessageBox.information(self, "Information", message)


class FrequencyRangeDialog(QDialog):
    """Dialog for entering frequency range for CUT file import."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        self.setWindowTitle("Frequency Range")
        self.setModal(True)
        
        layout = QFormLayout()
        
        # Frequency inputs (in GHz for user convenience)
        self.freq_start_spin = QDoubleSpinBox()
        self.freq_start_spin.setDecimals(3)
        self.freq_start_spin.setRange(0.001, 1000.0)
        self.freq_start_spin.setValue(2.0)
        self.freq_start_spin.setSuffix(" GHz")
        
        self.freq_end_spin = QDoubleSpinBox()
        self.freq_end_spin.setDecimals(3)
        self.freq_end_spin.setRange(0.001, 1000.0)
        self.freq_end_spin.setValue(18.0)
        self.freq_end_spin.setSuffix(" GHz")
        
        layout.addRow("Start Frequency:", self.freq_start_spin)
        layout.addRow("End Frequency:", self.freq_end_spin)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_frequency_range(self):
        """Get the frequency range in Hz."""
        freq_start_hz = self.freq_start_spin.value() * 1e9
        freq_end_hz = self.freq_end_spin.value() * 1e9
        return freq_start_hz, freq_end_hz