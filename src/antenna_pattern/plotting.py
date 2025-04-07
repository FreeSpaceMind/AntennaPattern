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

def plot_multiple_patterns(
    patterns: list,
    labels: list = None,
    colors: list = None,
    frequencies: list = None,
    phi_angles: list = None, 
    show_cross_pol: bool = False,
    value_type: str = 'gain',
    unwrap_phase: bool = True,
    title: str = None,
    ax = None,
    fig_size: tuple = (10, 6)
) -> tuple:
    """
    Plot multiple antenna patterns on the same axes with custom colors and labels.
    
    Args:
        patterns: List of AntennaPattern objects
        labels: List of legend labels for each pattern (defaults to Pattern 1, Pattern 2, etc.)
        colors: List of colors for each pattern (defaults to matplotlib default color cycle)
        frequencies: List of frequencies to plot for each pattern (or None to use first frequency of each)
        phi_angles: List of phi angles to plot for each pattern (or None to use first phi angle of each)
        show_cross_pol: If True, also plot cross-polarization components
        value_type: Type of value to plot ('gain', 'phase', or 'axial_ratio')
        unwrap_phase: If True and value_type is 'phase', unwrap phase to avoid 2π discontinuities
        title: Plot title
        ax: Optional matplotlib axes to plot on (created if None)
        fig_size: Figure size as (width, height) in inches
        
    Returns:
        tuple: (fig, ax) The figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.lines as mlines
    
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Set default labels if not provided
    if labels is None:
        labels = [f"Pattern {i+1}" for i in range(len(patterns))]
    
    # Normalize frequencies and phi_angles
    if frequencies is None:
        frequencies = [None] * len(patterns)
    elif len(frequencies) == 1 and len(patterns) > 1:
        frequencies = frequencies * len(patterns)
        
    if phi_angles is None:
        phi_angles = [None] * len(patterns)
    elif len(phi_angles) == 1 and len(patterns) > 1:
        phi_angles = phi_angles * len(patterns)
    
    # Make sure all input lists have the same length
    if not (len(patterns) == len(labels) == len(frequencies) == len(phi_angles)):
        raise ValueError("Input lists (patterns, labels, frequencies, phi_angles) must have compatible lengths")
    
    # Get default colors from matplotlib if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # Repeat colors if necessary
        if len(colors) < len(patterns):
            colors = colors * (len(patterns) // len(colors) + 1)
    elif len(colors) < len(patterns):
        raise ValueError(f"Not enough colors provided ({len(colors)}) for {len(patterns)} patterns")
    
    # Custom legend items
    legend_handles = []
    
    # Clear any existing legend
    if ax.get_legend():
        ax.get_legend().remove()
    
    # Process each pattern
    for i, (pattern, label, color, freq, phi) in enumerate(
            zip(patterns, labels, colors, frequencies, phi_angles)):
        
        # Get data arrays based on value_type from pattern
        theta_angles = pattern.theta_angles
        
        # Select frequency
        if freq is None:
            # Default to first frequency
            freq_value = pattern.frequencies[0]
            freq_idx = 0
        else:
            # Find nearest frequency
            from .utilities import find_nearest
            freq_value, freq_idx = find_nearest(pattern.frequencies, freq)
        
        # Select phi angles
        phi_angles_to_plot = phi if phi is not None else pattern.phi_angles
        
        # Get data for co-pol
        if value_type == 'gain':
            co_pol_data = pattern.get_gain_db('e_co').values[freq_idx]
            if show_cross_pol:
                cx_pol_data = pattern.get_gain_db('e_cx').values[freq_idx]
        elif value_type == 'phase':
            co_pol_data = pattern.get_phase('e_co', unwrapped=unwrap_phase).values[freq_idx]
            if show_cross_pol:
                cx_pol_data = pattern.get_phase('e_cx', unwrapped=unwrap_phase).values[freq_idx]
        elif value_type == 'axial_ratio':
            co_pol_data = pattern.get_axial_ratio().values[freq_idx]
            show_cross_pol = False  # No cross-pol for axial ratio
        else:
            raise ValueError(f"Invalid value_type: {value_type}")
        
        # Plot co-pol for each phi angle
        for phi_idx, phi_val in enumerate(phi_angles_to_plot):
            if phi_val not in pattern.phi_angles:
                # Find nearest phi angle
                from .utilities import find_nearest
                phi_val, phi_idx_actual = find_nearest(pattern.phi_angles, phi_val)
            else:
                phi_idx_actual = np.where(pattern.phi_angles == phi_val)[0][0]
            
            # Plot co-pol
            phi_label = f"{phi_val:.1f}°" if len(phi_angles_to_plot) > 1 else ""
            if phi_label:
                line_label = f"{label} (φ={phi_label})"
            else:
                line_label = label
                
            # Only first phi angle gets a label
            if phi_idx > 0:
                line_label = "_nolegend_"
                
            co_line = ax.plot(
                theta_angles, 
                co_pol_data[:, phi_idx_actual],
                '-',  # Solid line for co-pol
                color=color,
                label=line_label
            )[0]
            
            # Add to legend handles for first phi angle
            if phi_idx == 0:
                # Add to custom legend
                legend_handles.append(co_line)
            
            # Plot cross-pol if enabled
            if show_cross_pol:
                cx_line = ax.plot(
                    theta_angles,
                    cx_pol_data[:, phi_idx_actual],
                    '--',  # Dashed line for cross-pol
                    color=color,
                    label=f"{label} (cross-pol)" if phi_idx == 0 else "_nolegend_"
                )[0]
                
                # Add to legend handles for first phi angle
                if phi_idx == 0:
                    legend_handles.append(cx_line)
    
    # Set plot labels and grid
    ax.set_xlabel('Theta (degrees)')
    
    if value_type == 'gain':
        ax.set_ylabel('Gain (dBi)')
    elif value_type == 'phase':
        ax.set_ylabel('Phase (radians)')
    elif value_type == 'axial_ratio':
        ax.set_ylabel('Axial Ratio (linear)')
        
    # Set title if provided
    if title:
        ax.set_title(title)
        
    # Add grid
    ax.grid(True)
    
    # Add legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc='best')
    
    # Make layout tight
    fig.tight_layout()
    
    return fig, ax

def plot_pattern_difference(
    pattern1, 
    pattern2, 
    frequency: Optional[float] = None,
    phi: Optional[Union[float, List[float]]] = None,
    value_type: Literal['co_gain', 'cx_gain', 'axial_ratio', 'co_phase', 'cx_phase'] = 'co_gain',
    unwrap_phase: bool = True,
    ax: Optional[plt.Axes] = None,
    fig_size: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    absolute_diff: bool = True
) -> plt.Figure:
    """
    Plot the difference between two antenna patterns.
    
    Args:
        pattern1: First AntennaPattern object
        pattern2: Second AntennaPattern object
        frequency: Specific frequency to plot in Hz, or None to use first frequency
        phi: Specific phi angle(s) to plot in degrees, or None to use all matching phi angles
        value_type: Type of value to plot difference for:
            - 'co_gain': Co-polarized gain (dB)
            - 'cx_gain': Cross-polarized gain (dB)
            - 'axial_ratio': Axial ratio (linear)
            - 'co_phase': Co-polarized phase (radians)
            - 'cx_phase': Cross-polarized phase (radians)
        unwrap_phase: If True and value_type is phase, unwrap phase to avoid 2π discontinuities
        ax: Optional matplotlib axes to plot on
        fig_size: Figure size as (width, height) in inches
        title: Optional title for the plot
        absolute_diff: If True, plot absolute difference |p1-p2|, else plot signed difference (p1-p2)
        
    Returns:
        matplotlib.Figure: The created figure object
        
    Raises:
        ValueError: If patterns have incompatible dimensions
        ValueError: If value_type is invalid
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Validate value_type
    valid_types = ['co_gain', 'cx_gain', 'axial_ratio', 'co_phase', 'cx_phase']
    if value_type not in valid_types:
        raise ValueError(f"Invalid value_type: {value_type}. Must be one of {valid_types}")
    
    # Check if patterns have compatible dimensions
    if not np.array_equal(pattern1.theta_angles, pattern2.theta_angles):
        raise ValueError("Patterns have different theta angles. Cannot compute difference.")
    
    # Handle frequency selection
    if frequency is None:
        # Use first frequency from each pattern
        freq1_idx = 0
        freq2_idx = 0
        selected_freq1 = pattern1.frequencies[freq1_idx]
        selected_freq2 = pattern2.frequencies[freq2_idx]
    else:
        # Find nearest frequency in each pattern
        from .utilities import find_nearest
        selected_freq1, freq1_idx = find_nearest(pattern1.frequencies, frequency)
        selected_freq2, freq2_idx = find_nearest(pattern2.frequencies, frequency)
    
    # Handle phi selection
    if phi is None:
        # Use all matching phi angles between the two patterns
        common_phis = sorted(set(pattern1.phi_angles).intersection(set(pattern2.phi_angles)))
        if not common_phis:
            raise ValueError("Patterns have no common phi angles")
        
        # Create mapping from common phi values to indices in each pattern
        phi1_indices = [np.where(pattern1.phi_angles == p)[0][0] for p in common_phis]
        phi2_indices = [np.where(pattern2.phi_angles == p)[0][0] for p in common_phis]
        selected_phi = common_phis
    elif np.isscalar(phi):
        # Find nearest phi in each pattern
        from .utilities import find_nearest
        phi1_val, phi1_idx = find_nearest(pattern1.phi_angles, phi)
        phi2_val, phi2_idx = find_nearest(pattern2.phi_angles, phi)
        selected_phi = [phi1_val]  # Use phi from pattern1 for display
        phi1_indices = [phi1_idx]
        phi2_indices = [phi2_idx]
    else:
        # Multiple specific phi angles
        from .utilities import find_nearest
        phi1_indices = []
        phi2_indices = []
        selected_phi = []
        
        for p in phi:
            phi1_val, phi1_idx = find_nearest(pattern1.phi_angles, p)
            phi2_val, phi2_idx = find_nearest(pattern2.phi_angles, p)
            
            # Only use phis that are reasonably close to requested values
            if (abs(phi1_val - p) < 5.0) and (abs(phi2_val - p) < 5.0):
                phi1_indices.append(phi1_idx)
                phi2_indices.append(phi2_idx)
                selected_phi.append(phi1_val)
    
    # Determine y-label based on value_type
    if value_type == 'co_gain':
        y_label = 'Co-pol Gain Difference (dB)'
    elif value_type == 'cx_gain':
        y_label = 'Cross-pol Gain Difference (dB)'
    elif value_type == 'axial_ratio':
        y_label = 'Axial Ratio Difference'
    elif value_type == 'co_phase':
        y_label = 'Co-pol Phase Difference (rad)'
    elif value_type == 'cx_phase':
        y_label = 'Cross-pol Phase Difference (rad)'
    
    diff_label = 'Absolute Difference' if absolute_diff else 'Difference'
    
    # Get theta angles for x-axis
    theta_angles = pattern1.theta_angles
    
    # Plot for each phi angle
    for i, (phi1_idx, phi2_idx) in enumerate(zip(phi1_indices, phi2_indices)):
        phi_val = selected_phi[i]
        
        # Get data for this phi angle
        if value_type == 'co_gain':
            data1 = pattern1.get_gain_db('e_co')[freq1_idx, :, phi1_idx]
            data2 = pattern2.get_gain_db('e_co')[freq2_idx, :, phi2_idx]
        elif value_type == 'cx_gain':
            data1 = pattern1.get_gain_db('e_cx')[freq1_idx, :, phi1_idx]
            data2 = pattern2.get_gain_db('e_cx')[freq2_idx, :, phi2_idx]
        elif value_type == 'axial_ratio':
            data1 = pattern1.get_axial_ratio()[freq1_idx, :, phi1_idx]
            data2 = pattern2.get_axial_ratio()[freq2_idx, :, phi2_idx]
        elif value_type == 'co_phase':
            data1 = pattern1.get_phase('e_co', unwrapped=unwrap_phase)[freq1_idx, :, phi1_idx]
            data2 = pattern2.get_phase('e_co', unwrapped=unwrap_phase)[freq2_idx, :, phi2_idx]
        elif value_type == 'cx_phase':
            data1 = pattern1.get_phase('e_cx', unwrapped=unwrap_phase)[freq1_idx, :, phi1_idx]
            data2 = pattern2.get_phase('e_cx', unwrapped=unwrap_phase)[freq2_idx, :, phi2_idx]
        
        # Calculate difference
        if absolute_diff:
            difference = np.abs(data1 - data2)
        else:
            difference = data1 - data2
        
        # Plot difference - no labels for legend
        ax.plot(theta_angles, difference)
    
    # Set plot labels and grid
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel(y_label)
    
    # Create title if not provided
    if title is None:
        freq_str = f"{selected_freq1/1e6:.1f} MHz"
        
        if len(selected_phi) == 1:
            phi_str = f"φ={selected_phi[0]:.1f}°"
        else:
            phi_str = f"{len(selected_phi)} φ cuts"
        
        title = f"Pattern {diff_label}: {freq_str}, {phi_str}"
    
    ax.set_title(title)
    ax.grid(True)
    
    # Make layout tight
    fig.tight_layout()
    
    return fig