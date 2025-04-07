"""
Plotting functions for antenna radiation patterns.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Literal
import logging

from .pattern import AntennaPattern
from .utilities import find_nearest

def plot_pattern_cut(
    pattern: AntennaPattern,
    frequency: Optional[float] = None,
    phi: Optional[Union[float, List[float]]] = None,
    show_cross_pol: bool = True,
    value_type: Literal['gain', 'phase', 'axial_ratio'] = 'gain',
    unwrap_phase: bool = True,
    ax: Optional[plt.Axes] = None,
    fig_size: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot antenna pattern cuts with selectable value type.
    
    Args:
        pattern: AntennaPattern object
        frequency: Specific frequency to plot in Hz, or None to use first frequency
        phi: Specific phi angle(s) to plot in degrees, or None to use all phi
        show_cross_pol: If True, plot both co-pol and cross-pol components (ignored for axial_ratio)
        value_type: Type of value to plot ('gain', 'phase', or 'axial_ratio')
        unwrap_phase: If True and value_type is 'phase', unwrap phase to avoid 2π discontinuities
        ax: Optional matplotlib axes to plot on
        fig_size: Figure size as (width, height) in inches
        title: Optional title for the plot
        
    Returns:
        matplotlib.Figure: The created figure object
    """
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Handle frequency selection
    frequencies = pattern.frequencies
    if frequency is None:
        frequency_indices = [0]  # Default to first frequency
        selected_frequencies = [frequencies[0]]
    elif np.isscalar(frequency):
        # Find nearest frequency
        nearest_freq, freq_idx = find_nearest(frequencies, frequency)
        frequency_indices = [freq_idx]
        selected_frequencies = [nearest_freq]
    else:
        # Multiple frequencies
        frequency_indices = []
        selected_frequencies = []
        for f in frequency:
            nearest_freq, freq_idx = find_nearest(frequencies, f)
            frequency_indices.append(freq_idx)
            selected_frequencies.append(nearest_freq)
    
    # Handle phi selection
    phi_angles = pattern.phi_angles
    if phi is None:
        phi_indices = list(range(len(phi_angles)))
        selected_phi = phi_angles
    elif np.isscalar(phi):
        # Find nearest phi
        nearest_phi, phi_idx = find_nearest(phi_angles, phi)
        phi_indices = [phi_idx]
        selected_phi = [nearest_phi]
    else:
        # Multiple phi angles
        phi_indices = []
        selected_phi = []
        for p in phi:
            nearest_phi, phi_idx = find_nearest(phi_angles, p)
            phi_indices.append(phi_idx)
            selected_phi.append(nearest_phi)
    
    # Get data arrays based on value_type
    theta_angles = pattern.theta_angles
    
    # Special case for axial ratio - no cross-pol plotting
    if value_type == 'axial_ratio':
        show_cross_pol = False
        data_co = pattern.get_axial_ratio()
        data_cx = None
        y_label = 'Axial Ratio (linear)'
        plot_prefix = 'AR'
    elif value_type == 'phase':
        data_co = pattern.get_phase('e_co', unwrapped=unwrap_phase)
        data_cx = pattern.get_phase('e_cx', unwrapped=unwrap_phase) if show_cross_pol else None
        y_label = 'Phase (radians)'
        plot_prefix = 'Phase'
    else:  # Default to gain
        data_co = pattern.get_gain_db('e_co')
        data_cx = pattern.get_gain_db('e_cx') if show_cross_pol else None
        y_label = 'Gain (dBi)'
        plot_prefix = 'Gain'
    
    # Determine total number of lines to plot
    num_lines = len(frequency_indices) * len(phi_indices) * (2 if show_cross_pol else 1)
    
    # Define line styles
    co_pol_style = '-'
    cx_pol_style = ':'
    
    # Define color mappings
    # If more than 8 lines, use a color per (frequency, polarization) group
    if num_lines > 8:
        # Use color cycle for frequencies
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(frequency_indices)))
        
        # Plot with frequency-grouped colors
        for i, freq_idx in enumerate(frequency_indices):
            freq_value = selected_frequencies[i]
            color = color_cycle[i % len(color_cycle)]
            
            # Group by frequency - plot all phi angles with same color
            for phi_idx in phi_indices:
                # Plot co-pol
                if value_type == 'axial_ratio':
                    label = f"{freq_value/1e6:.1f} MHz" if phi_idx == phi_indices[0] else None
                else:
                    label = f"{freq_value/1e6:.1f} MHz (co-pol)" if phi_idx == phi_indices[0] else None
                
                ax.plot(
                    theta_angles,
                    data_co.values[freq_idx, :, phi_idx],
                    co_pol_style,
                    color=color,
                    alpha=0.8,
                    label=label
                )
                
                # Plot cross-pol
                if show_cross_pol and data_cx is not None:
                    ax.plot(
                        theta_angles,
                        data_cx.values[freq_idx, :, phi_idx],
                        cx_pol_style,
                        color=color,
                        alpha=0.8,
                        label=f"{freq_value/1e6:.1f} MHz (cross-pol)" if phi_idx == phi_indices[0] else None
                    )
    else:
        # Less than 8 lines, use a color per phi angle
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(phi_indices)))
        
        # Plot with detailed labels
        for i, freq_idx in enumerate(frequency_indices):
            freq_value = selected_frequencies[i]
            
            for j, phi_idx in enumerate(phi_indices):
                phi_value = selected_phi[j]
                color = color_cycle[j % len(color_cycle)]
                
                # Create labels based on value type and frequencies
                if len(frequency_indices) > 1:
                    label = f"φ={phi_value:.1f}°, f={freq_value/1e6:.1f} MHz"
                else:
                    label = f"φ={phi_value:.1f}°"
                
                # Plot co-pol
                ax.plot(
                    theta_angles,
                    data_co.values[freq_idx, :, phi_idx],
                    co_pol_style,
                    color=color,
                    label=label
                )
                
                # Plot cross-pol if enabled and not axial ratio
                if show_cross_pol and data_cx is not None:
                    if len(frequency_indices) > 1:
                        cx_label = f"φ={phi_value:.1f}°, f={freq_value/1e6:.1f} MHz (cross)"
                    else:
                        cx_label = f"φ={phi_value:.1f}° (cross)"
                    
                    ax.plot(
                        theta_angles,
                        data_cx.values[freq_idx, :, phi_idx],
                        cx_pol_style,
                        color=color,
                        label=cx_label
                    )
    
    # Set plot labels and grid
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel(y_label)
    
    # Create title if not provided
    if title is None:
        if len(frequency_indices) == 1:
            freq_str = f"{selected_frequencies[0]/1e6:.1f} MHz"
        else:
            freq_str = "Multiple Frequencies"
        
        if len(phi_indices) == 1:
            phi_str = f"φ={selected_phi[0]:.1f}°"
        else:
            phi_str = "Multiple φ Cuts"
        
        title = f"Antenna {plot_prefix} Pattern: {freq_str}, {phi_str}"
    
    ax.set_title(title)
    ax.grid(True)
    
    # Add legend
    if num_lines > 0:
        ax.legend(loc='best')
    
    # Make layout tight
    fig.tight_layout()
    
    return fig