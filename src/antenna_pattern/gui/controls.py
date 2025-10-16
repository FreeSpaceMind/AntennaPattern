"""
Main controls widget with tabbed interface.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtCore import pyqtSignal

from .view_tab import ViewTab
from .processing_tab import ProcessingTab
from .analysis_tab import AnalysisTab
from .swe_worker import SWEWorker


class ControlsWidget(QWidget):
    """Main controls widget containing tabbed interface."""
    
    # Relay signals from tabs
    parameters_changed = pyqtSignal()
    apply_phase_center_signal = pyqtSignal(float, float, float, float)
    apply_mars_signal = pyqtSignal(float)
    calculate_swe_signal = pyqtSignal()
    calculate_nearfield_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pattern = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the tabbed control interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.view_tab = ViewTab()
        self.processing_tab = ProcessingTab()
        self.analysis_tab = AnalysisTab()
        
        # Add tabs
        self.tab_widget.addTab(self.view_tab, "View")
        self.tab_widget.addTab(self.processing_tab, "Processing")
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        
        # Connect signals from tabs
        self.view_tab.parameters_changed.connect(self.parameters_changed.emit)
        self.processing_tab.apply_phase_center_signal.connect(self.apply_phase_center_signal.emit)
        self.processing_tab.apply_mars_signal.connect(self.apply_mars_signal.emit)
        self.processing_tab.coordinate_format_changed.connect(self.on_coordinate_format_changed)
        # Note: polarization_changed should be connected directly in main_window
        # to on_polarization_changed, not relayed through parameters_changed
        self.analysis_tab.calculate_swe_signal.connect(self.on_calculate_swe)
        self.analysis_tab.calculate_nearfield_signal.connect(self.on_calculate_nearfield)

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def update_pattern(self, pattern):
        """Update all tabs with new pattern."""
        self.current_pattern = pattern
        self.view_tab.update_pattern(pattern)
        self.processing_tab.update_pattern(pattern)
        self.analysis_tab.update_pattern(pattern)
    
    def reset_processing_state(self):
        """Reset processing checkboxes when loading new pattern."""
        self.processing_tab.reset_processing_state()
    
    def on_calculate_swe(self):
        """Handle SWE calculation request."""
        if not self.current_pattern:
            return
        
        # Prevent multiple simultaneous calculations
        if hasattr(self, 'swe_worker') and self.swe_worker.isRunning():
            return
        
        try:
            # Import the worker
            from .swe_worker import SWEWorker
            
            # Get parameters
            adaptive = self.analysis_tab.get_swe_adaptive()
            radius = None if adaptive else self.analysis_tab.get_swe_radius()
            frequency = self.analysis_tab.get_swe_frequency()
            
            # Update button state
            self.analysis_tab.calculate_swe_btn.setEnabled(False)
            self.analysis_tab.calculate_swe_btn.setText("Calculating...")
            
            # Create and configure worker thread
            self.swe_worker = SWEWorker(
                self.current_pattern,
                radius,
                frequency,
                adaptive
            )
            
            # Connect signals
            self.swe_worker.finished.connect(self.on_swe_finished)
            self.swe_worker.error.connect(self.on_swe_error)
            self.swe_worker.progress.connect(self.on_swe_progress)
            
            # Start the calculation in background
            self.swe_worker.start()
            
        except Exception as e:
            import logging
            logging.error(f"Error starting SWE calculation: {e}", exc_info=True)
            self.analysis_tab.swe_results.setText(f"Error: {str(e)}")
            self.analysis_tab.calculate_swe_btn.setEnabled(True)
            self.analysis_tab.calculate_swe_btn.setText("Calculate SWE Coefficients")
    
    def on_swe_finished(self, swe_data):
        """Handle successful SWE calculation."""
        # Display results
        self.analysis_tab.display_swe_results(swe_data)
        
        # Re-enable button
        self.analysis_tab.calculate_swe_btn.setEnabled(True)
        self.analysis_tab.calculate_swe_btn.setText("Calculate SWE Coefficients")
    
    def on_swe_error(self, error_msg):
        """Handle SWE calculation error."""
        self.analysis_tab.swe_results.setText(f"Error: {error_msg}")
        
        # Re-enable button
        self.analysis_tab.calculate_swe_btn.setEnabled(True)
        self.analysis_tab.calculate_swe_btn.setText("Calculate SWE Coefficients")
    
    def on_swe_progress(self, message):
        """Handle SWE calculation progress updates."""
        # Could update a progress bar or status message here
        pass

    def on_coordinate_format_changed(self, format_type):
        """Handle coordinate format change."""
        if self.current_pattern:
            try:
                self.current_pattern.transform_coordinates(format_type)
                # Update all tabs with the transformed pattern
                self.view_tab.update_pattern(self.current_pattern)
                self.processing_tab.update_pattern(self.current_pattern)
                self.analysis_tab.update_pattern(self.current_pattern)
                self.parameters_changed.emit()
            except Exception as e:
                import logging
                logging.error(f"Error transforming coordinates: {e}", exc_info=True)
    
    def on_calculate_nearfield(self):
        """Handle near field calculation request."""
        if not self.current_pattern:
            return
        
        try:
            import numpy as np
            surface_type = self.analysis_tab.get_nf_surface_type()
            
            if surface_type == "spherical":
                # Get spherical parameters
                params = self.analysis_tab.get_nf_sphere_params()
                
                # Create theta and phi arrays
                theta = np.linspace(0, 180, params['theta_points'])
                phi = np.linspace(0, 360, params['phi_points'])
                
                # Evaluate near field
                nf_data = self.current_pattern.evaluate_nearfield_sphere(
                    radius=params['radius'],
                    theta_points=theta,
                    phi_points=phi
                )
                
                nf_data['is_spherical'] = True
                
            else:  # planar
                # Get planar parameters
                params = self.analysis_tab.get_nf_plane_params()
                
                # Create x and y arrays
                x = np.linspace(-params['x_extent'], params['x_extent'], params['x_points'])
                y = np.linspace(-params['y_extent'], params['y_extent'], params['y_points'])
                
                # Evaluate near field
                nf_data = self.current_pattern.evaluate_nearfield_plane(
                    x_points=x,
                    y_points=y,
                    z_plane=params['z_distance']
                )
                
                nf_data['is_spherical'] = False
                nf_data['x_extent'] = params['x_extent']
                nf_data['y_extent'] = params['y_extent']
                nf_data['z_distance'] = params['z_distance']
            
            # Display results
            self.analysis_tab.display_nearfield_results(nf_data)
            
            # Open near field viewer window
            from .nearfield_viewer import NearFieldViewer
            viewer = NearFieldViewer(nf_data, parent=self)
            viewer.show()
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.analysis_tab.nf_results.setText(error_msg)
    
    # Getter methods - delegate to appropriate tabs
    def get_selected_frequencies(self):
        """Get selected frequencies from view tab."""
        return self.view_tab.get_selected_frequencies()
    
    def get_selected_phi_angles(self):
        """Get selected phi angles from view tab."""
        return self.view_tab.get_selected_phi_angles()
    
    def get_plot_format(self):
        """Get plot format from view tab."""
        return self.view_tab.get_plot_format()
    
    def get_value_type(self):
        """Get value type from view tab."""
        return self.view_tab.get_value_type()
    
    def get_component(self):
        """Get component from view tab."""
        return self.view_tab.get_component()
    
    def get_show_cross_pol(self):
        """Get show cross-pol state from view tab."""
        return self.view_tab.get_show_cross_pol()
    
    def get_statistics_enabled(self):
        """Get statistics enabled state from view tab."""
        return self.view_tab.get_statistics_enabled()
    
    def get_show_range(self):
        """Get show range state from view tab."""
        return self.view_tab.get_show_range()
    
    def get_statistic_type(self):
        """Get statistic type from view tab."""
        return self.view_tab.get_statistic_type()
    
    def get_percentile_range(self):
        """Get percentile range from view tab."""
        return self.view_tab.get_percentile_range()
    
    def get_polarization(self):
        """Get polarization from processing tab."""
        return self.processing_tab.get_polarization()
    
    # For backward compatibility with existing code
    @property
    def apply_phase_center_check(self):
        """Access phase center checkbox."""
        return self.processing_tab.apply_phase_center_check
    
    @property
    def apply_mars_check(self):
        """Access MARS checkbox."""
        return self.processing_tab.apply_mars_check
    
    @property
    def max_radial_extent_spin(self):
        """Access max radial extent spinbox."""
        return self.processing_tab.max_radial_extent_spin
    
    def update_controls_for_plot_format(self):
        """Update control visibility based on plot format."""
        # This method is called from main_window but might not be needed anymore
        # Keep for compatibility but can be removed later
        pass