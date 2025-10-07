"""
Plotting functions for antenna radiation patterns.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Literal, Any, Dict

from .pattern import AntennaPattern
from .utilities import find_nearest

def plot_pattern_cut(
    pattern: AntennaPattern,
    frequency: Optional[float] = None,
    phi: Optional[Union[float, List[float]]] = None,
    show_cross_pol: bool = True,
    value_type: Literal['gain', 'phase', 'axial_ratio'] = 'gain',
    unwrap_phase: bool = True,
    normalize: bool = False, 
    component: str = 'e_co',
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
        normalize: If True, normalize peak to 0 dB
        component: Component to plot ('e_co', 'e_cx', 'e_theta', 'e_phi') 
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

    # Determine which component to use
    # For e_co, cross-pol is e_cx; for e_theta, cross-pol is e_phi, etc.
    if component == 'e_co':
        main_component = 'e_co'
        cross_component = 'e_cx'
    elif component == 'e_cx':
        main_component = 'e_cx'
        cross_component = 'e_co'
    elif component == 'e_theta':
        main_component = 'e_theta'
        cross_component = 'e_phi'
    elif component == 'e_phi':
        main_component = 'e_phi'
        cross_component = 'e_theta'
    else:
        # Default to co-pol
        main_component = 'e_co'
        cross_component = 'e_cx'

    # Special case for axial ratio - no cross-pol plotting
    if value_type == 'axial_ratio':
        show_cross_pol = False
        data_co = pattern.get_axial_ratio()
        data_cx = None
        y_label = 'Axial Ratio (dB)'
        plot_prefix = 'AR'
    elif value_type == 'phase':
        data_co = pattern.get_phase(main_component, unwrapped=unwrap_phase)
        data_cx = pattern.get_phase(cross_component, unwrapped=unwrap_phase) if show_cross_pol else None
        y_label = 'Phase (degrees)'
        plot_prefix = 'Phase'
    else:  # Default to gain
        data_co = pattern.get_gain_db(main_component)
        data_cx = pattern.get_gain_db(cross_component) if show_cross_pol else None
        y_label = 'Gain (dBi)'
        plot_prefix = 'Gain'

    # normalize if requested
    if normalize and value_type == 'gain':
        # Normalize peak gain to zero
        peak_value = np.max(data_co[frequency_indices, :, :][:, :, phi_indices])
        data_co = data_co - peak_value
        if data_cx is not None:
            data_cx = data_cx - peak_value
    elif normalize and value_type == 'phase':
        # Normalize first cut's phase to zero
        if phi_indices:
            ref_phi_idx = phi_indices[0]
            ref_freq_idx = frequency_indices[0]
            ref_phase = data_co[ref_freq_idx, 0, ref_phi_idx]  # First theta point
            data_co = data_co - ref_phase
            if data_cx is not None:
                data_cx = data_cx - ref_phase
    
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
                    data_co[freq_idx, :, phi_idx],
                    co_pol_style,
                    color=color,
                    alpha=0.8,
                    label=label
                )
                
                # Plot cross-pol
                if show_cross_pol and data_cx is not None:
                    ax.plot(
                        theta_angles,
                        data_cx[freq_idx, :, phi_idx],
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
                    data_co[freq_idx, :, phi_idx],
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
                        data_cx[freq_idx, :, phi_idx],
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
    normalize_phase: bool = True,
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
            co_pol_data = pattern.get_gain_db('e_co')[freq_idx]
            if show_cross_pol:
                cx_pol_data = pattern.get_gain_db('e_cx')[freq_idx]
        elif value_type == 'phase':
            co_pol_data = pattern.get_phase('e_co', unwrapped=unwrap_phase)[freq_idx]
            if show_cross_pol:
                cx_pol_data = pattern.get_phase('e_cx', unwrapped=unwrap_phase)[freq_idx]

            # Apply phase normalization if requested
            if normalize_phase is not False:
                # boresight (theta=0) and first phi angle
                ref_theta_idx = np.argmin(np.abs(theta_angles))
                ref_phi_idx = 0
                
                # Get reference phase at the specified point
                ref_phase_co = co_pol_data[ref_theta_idx, ref_phi_idx]
                
                # Normalize co-pol data
                co_pol_data = co_pol_data - ref_phase_co
                
                # Normalize cross-pol data if present
                if show_cross_pol:
                    ref_phase_cx = cx_pol_data[ref_theta_idx, ref_phi_idx]
                    cx_pol_data = cx_pol_data - ref_phase_cx

        elif value_type == 'axial_ratio':
            co_pol_data = pattern.get_axial_ratio()[freq_idx]
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
        ax.set_ylabel('Phase (degrees)')
    elif value_type == 'axial_ratio':
        ax.set_ylabel('Axial Ratio (dB)')
        
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
            - 'axial_ratio': Axial ratio (dB)
            - 'co_phase': Co-polarized phase (degrees)
            - 'cx_phase': Cross-polarized phase (degrees)
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
        y_label = 'Axial Ratio Difference (dB)'
    elif value_type == 'co_phase':
        y_label = 'Co-pol Phase Difference (deg)'
    elif value_type == 'cx_phase':
        y_label = 'Cross-pol Phase Difference (deg)'
    
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

def plot_pattern_statistics(
    pattern: AntennaPattern,
    statistic_over: Literal['phi', 'frequency'] = 'phi',
    frequency: Optional[Union[float, List[float]]] = None,
    phi: Optional[Union[float, List[float]]] = None,
    component: str = 'e_co',
    value_type: Literal['gain', 'phase', 'axial_ratio'] = 'gain',
    statistic: Literal['mean', 'median', 'rms', 'percentile', 'std'] = 'mean',
    percentile_range: Tuple[float, float] = (25, 75),
    show_range: bool = True,
    ax: Optional[plt.Axes] = None,
    fig_size: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    show_individual: bool = False,
    alpha_individual: float = 0.2,
    legend: bool = True
) -> plt.Figure:
    """
    Plot statistical measures across phi angles or frequencies for a single pattern.
    
    Args:
        pattern: AntennaPattern object
        statistic_over: Dimension to compute statistics over ('phi' or 'frequency')
        frequency: Specific frequency or list of frequencies to use (in Hz).
                   If None and statistic_over='phi', the first frequency is used.
                   If statistic_over='frequency', this parameter is ignored.
        phi: Specific phi angle or list of phi angles to use (in degrees).
             If None and statistic_over='frequency', the first phi is used.
             If statistic_over='phi', this parameter is ignored.
        component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')
        value_type: Type of value to plot ('gain', 'phase', or 'axial_ratio')
        statistic: Statistical measure to plot:
            - 'mean': Arithmetic mean
            - 'median': Median value
            - 'rms': Root Mean Square value
            - 'percentile': Show percentile range (controlled by percentile_range)
            - 'std': Mean plus/minus one standard deviation
        percentile_range: Tuple of (lower, upper) percentiles to show when statistic='percentile'
        show_range: If True, show min/max range as shaded area
        ax: Optional matplotlib axes to plot on
        fig_size: Figure size as (width, height) in inches
        title: Optional title for the plot
        show_individual: If True, plot individual cuts as thin lines
        alpha_individual: Alpha transparency for individual lines
        legend: If True, add a legend to the plot
        
    Returns:
        matplotlib.Figure: The created figure object
        
    Raises:
        ValueError: If statistics type is invalid
    """
    # Validate statistic type
    valid_stats = ['mean', 'median', 'rms', 'percentile', 'std']
    if statistic not in valid_stats:
        raise ValueError(f"Invalid statistic: {statistic}. Must be one of {valid_stats}")
    
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Get theta angles for the x-axis
    theta = pattern.theta_angles
    
    # Initialize data based on statistic_over dimension
    if statistic_over == 'phi':
        # Get data across all phi angles
        if frequency is None:
            freq_idx = 0
            selected_frequency = pattern.frequencies[0]
        elif isinstance(frequency, (list, tuple, np.ndarray)):
            # Multiple frequencies - use first one
            nearest_freq, freq_idx = find_nearest(pattern.frequencies, frequency[0])
            selected_frequency = nearest_freq
        else:
            # Single frequency
            nearest_freq, freq_idx = find_nearest(pattern.frequencies, frequency)
            selected_frequency = nearest_freq
        
        # Get all phi angles
        phi_angles = pattern.phi_angles
        
        # Extract data based on value_type
        if value_type == 'gain':
            # Shape: [theta, phi]
            data = pattern.get_gain_db(component)[freq_idx]
        elif value_type == 'phase':
            data = pattern.get_phase(component, unwrapped=True)[freq_idx]
        elif value_type == 'axial_ratio':
            data = pattern.get_axial_ratio()[freq_idx]
        else:
            raise ValueError(f"Invalid value_type: {value_type}")
            
        dimension_label = f"φ angles ({len(phi_angles)})"
        dimension_values = phi_angles
        
    elif statistic_over == 'frequency':
        # Get data across all frequencies
        if phi is None:
            phi_idx = 0
            selected_phi = pattern.phi_angles[0]
        elif isinstance(phi, (list, tuple, np.ndarray)):
            # Multiple phi angles - use first one
            nearest_phi, phi_idx = find_nearest(pattern.phi_angles, phi[0])
            selected_phi = nearest_phi
        else:
            # Single phi angle
            nearest_phi, phi_idx = find_nearest(pattern.phi_angles, phi)
            selected_phi = nearest_phi
        
        # Get all frequencies
        frequencies = pattern.frequencies
        
        # Extract data based on value_type
        if value_type == 'gain':
            # Shape: [freq, theta]
            data = pattern.get_gain_db(component)[:, :, phi_idx]
        elif value_type == 'phase':
            data = pattern.get_phase(component, unwrapped=True)[:, :, phi_idx]
        elif value_type == 'axial_ratio':
            data = pattern.get_axial_ratio()[:, :, phi_idx]
        else:
            raise ValueError(f"Invalid value_type: {value_type}")
            
        dimension_label = f"Frequencies ({len(frequencies)})"
        dimension_values = frequencies / 1e9  # Convert to GHz for display
        
    else:
        raise ValueError(f"Invalid statistic_over: {statistic_over}. Must be 'phi' or 'frequency'")
    
    # Transpose data to have shape [dimension, theta]
    # When statistic_over='phi', dimension represents different phi angles
    # When statistic_over='frequency', dimension represents different frequencies
    all_data = data.T if statistic_over == 'phi' else data
    
    # Calculate statistics across the dimension (phi or frequency)
    mean_data = np.mean(all_data, axis=0)
    median_data = np.median(all_data, axis=0)
    std_data = np.std(all_data, axis=0)
    min_data = np.min(all_data, axis=0)
    max_data = np.max(all_data, axis=0)
    
    # For RMS, we need to square values, mean, then sqrt
    rms_data = np.sqrt(np.mean(all_data**2, axis=0))

    
    # Calculate percentiles if needed
    lower_percentile = np.percentile(all_data, percentile_range[0], axis=0)
    upper_percentile = np.percentile(all_data, percentile_range[1], axis=0)
    
    # Plot individual cuts if requested
    if show_individual:
        for i in range(len(all_data)):
            if statistic_over == 'phi':
                label = f"φ={dimension_values[i]:.1f}°"
            else:
                label = f"{dimension_values[i]:.2f} GHz"
                
            # Only add to legend if there aren't too many lines
            if len(all_data) > 10:
                label = "_nolegend_"
                
            ax.plot(theta, all_data[i], 
                   alpha=alpha_individual, 
                   linewidth=0.8,
                   label=label)
    
    # Set plot color and label
    stat_color = 'blue'
    range_alpha = 0.2
    
    # Plot the requested statistic
    if statistic == 'mean':
        ax.plot(theta, mean_data, color=stat_color, linewidth=2, 
               label=f"Mean across {dimension_label}")
        stat_label = 'Mean'
        
    elif statistic == 'median':
        ax.plot(theta, median_data, color=stat_color, linewidth=2, 
               label=f"Median across {dimension_label}")
        stat_label = 'Median'
        
    elif statistic == 'rms':
        ax.plot(theta, rms_data, color=stat_color, linewidth=2, 
               label=f"RMS across {dimension_label}")
        stat_label = 'RMS'
        
    elif statistic == 'percentile':
        ax.plot(theta, median_data, color=stat_color, linewidth=2, 
               label=f"Median across {dimension_label}")
        ax.fill_between(theta, lower_percentile, upper_percentile, alpha=range_alpha, 
                       color=stat_color, 
                       label=f"{percentile_range[0]}-{percentile_range[1]} Percentile")
        stat_label = f"Percentile Range {percentile_range[0]}-{percentile_range[1]}"
        
    elif statistic == 'std':
        ax.plot(theta, mean_data, color=stat_color, linewidth=2, 
               label=f"Mean across {dimension_label}")
        ax.fill_between(theta, mean_data - std_data, mean_data + std_data, alpha=range_alpha, 
                       color=stat_color, label=f"±1 Std Dev")
        stat_label = 'Mean ±1 Std Dev'
    
    # Show min/max range if requested
    if show_range and statistic != 'percentile':  # Don't add min/max if already showing percentiles
        ax.fill_between(theta, min_data, max_data, alpha=range_alpha/2, color='gray', 
                       label="Min-Max Range")
    
    # Set plot labels and grid
    ax.set_xlabel('Theta (degrees)')
    
    if value_type == 'gain':
        y_label = f'{component.lower()} Gain (dBi)'
    elif value_type == 'phase':
        y_label = f'{component.lower()} Phase (degrees)'
    elif value_type == 'axial_ratio':
        y_label = 'Axial Ratio (dB)'
    
    ax.set_ylabel(y_label)
    
    # Create title if not provided
    if title is None:
        if statistic_over == 'phi':
            value_str = f"{selected_frequency/1e6:.1f} MHz"
            over_str = "φ angles"
        else:
            value_str = f"φ={selected_phi:.1f}°"
            over_str = "frequencies"
            
        title = f"{stat_label} {value_type.capitalize()} across {over_str}: {value_str}"
    
    ax.set_title(title)
    ax.grid(True)
    
    # Add legend if requested
    if legend:
        ax.legend(loc='best')
    
    # Make layout tight
    fig.tight_layout()
    
    return fig

def add_spec_mask(
    ax: plt.Axes,
    x_points: Union[List[float], np.ndarray],
    y_points: Union[List[float], np.ndarray],
    mask_type: Literal['upper', 'lower', 'both'] = 'upper',
    fill: bool = True,
    color: Optional[Union[str, Tuple[float, float, float]]] = None,
    line_style: str = '-',
    line_width: float = 2.0,
    fill_alpha: float = 0.2,
    label: Optional[str] = None,
) -> List[Any]:
    """
    Add a specification mask or line to an existing antenna pattern plot.
    
    Args:
        ax: Matplotlib axes to add the spec mask to
        x_points: X-axis points defining the spec mask (typically theta angles in degrees)
        y_points: Y-axis points defining the spec mask (typically gain values in dB)
        mask_type: Type of specification mask:
            - 'upper': Mask is an upper limit (highlight region above the line)
            - 'lower': Mask is a lower limit (highlight region below the line)
            - 'both': Mask is a boundary line only (no fill)
        fill: If True, fill the mask region
        color: Color for the mask line and fill; if None, uses:
            - Red for upper limits
            - Green for lower limits
            - Blue for boundary lines
        line_style: Style for the mask line ('solid', 'dashed', etc.)
        line_width: Width of the mask line
        fill_alpha: Alpha transparency for the filled region
        label: Label for the mask line in the legend
        
    Returns:
        List of artists added to the plot (can be used for legend entries)
        
    Notes:
        - For a proper envelope mask with segments, provide all corner points in the mask
    """
    from scipy import interpolate
    
    # Convert inputs to numpy arrays
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)
    
    # Validate inputs
    if len(x_points) != len(y_points):
        raise ValueError(f"x_points length ({len(x_points)}) must match y_points length ({len(y_points)})")
    
    if len(x_points) < 2:
        raise ValueError("At least two points are required to define a spec mask")
    
    # Set default colors based on mask_type if not provided
    if color is None:
        if mask_type == 'upper':
            color = 'red'
        elif mask_type == 'lower':
            color = 'green'
        else:  # 'both'
            color = 'blue'
    
    # Create list to store added artists
    artists = []
    
    # Plot the spec line
    line = ax.plot(
        x_points, 
        y_points, 
        linestyle=line_style, 
        linewidth=line_width, 
        color=color,
        label=label
    )[0]
    artists.append(line)
    
    # Add fill if requested
    if fill and mask_type != 'both':
        # Get current y-axis limits to determine fill boundaries
        y_min, y_max = ax.get_ylim()
        
        if mask_type == 'upper':
            # Fill from the line up to the top of the plot
            fill_artist = ax.fill_between(
                x_points,
                y_points,
                np.full_like(y_points, y_max),
                color=color,
                alpha=fill_alpha,
                label='_nolegend_'  # Don't add fill to legend
            )
        else:  # 'lower'
            # Fill from the line down to the bottom of the plot
            fill_artist = ax.fill_between(
                x_points,
                np.full_like(y_points, y_min),
                y_points,
                color=color,
                alpha=fill_alpha,
                label='_nolegend_'  # Don't add fill to legend
            )
        
        artists.append(fill_artist)
    
    return artists


def add_envelope_spec(
    ax: plt.Axes,
    spec_points: Dict[str, List[Tuple[float, float]]],
    colors: Optional[Dict[str, str]] = None,
    fill: bool = True,
    fill_alpha: float = 0.2,
    line_width: float = 2.0,
) -> Dict[str, List[Any]]:
    """
    Add multiple specification masks to create an envelope specification.
    
    Args:
        ax: Matplotlib axes to add the spec masks to
        spec_points: Dictionary mapping spec names to lists of (x, y) tuples defining each spec
        colors: Optional dictionary mapping spec names to colors
        fill: If True, fill the mask regions
        fill_alpha: Alpha transparency for the filled regions
        line_width: Width of the mask lines
        
    Returns:
        Dictionary mapping spec names to lists of artists added to the plot
        
    Example:
        ```python
        # Define spec points
        spec_points = {
            'upper_limit': [(0, 20), (10, 15), (30, 5), (90, 0)],  # Upper limit spec
            'lower_limit': [(0, 10), (10, 8), (30, -5), (90, -10)]  # Lower limit spec
        }
        
        # Add specs to plot
        artists = add_envelope_spec(ax, spec_points, fill=True)
        ```
    """
    # Default colors if not provided
    if colors is None:
        colors = {
            'upper': 'red',
            'lower': 'green',
            'target': 'blue'
        }
    
    # Create default color mapping for any spec names
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Process each spec and add to plot
    result_artists = {}
    
    for i, (spec_name, points) in enumerate(spec_points.items()):
        # Split points into x and y arrays
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        
        # Determine if this is an upper, lower or target spec based on name
        if 'upper' in spec_name.lower() or 'max' in spec_name.lower():
            mask_type = 'upper'
        elif 'lower' in spec_name.lower() or 'min' in spec_name.lower():
            mask_type = 'lower'
        else:
            mask_type = 'both'  # Default to boundary line only
        
        # Get color for this spec
        if spec_name in colors:
            color = colors[spec_name]
        elif mask_type in colors:
            color = colors[mask_type]
        else:
            # Use next color from default cycle
            color = default_colors[i % len(default_colors)]
        
        # Add the spec mask
        artists = add_spec_mask(
            ax=ax,
            x_points=x_points,
            y_points=y_points,
            mask_type=mask_type,
            fill=fill,
            color=color,
            line_width=line_width,
            fill_alpha=fill_alpha,
            label=spec_name,
        )
        
        result_artists[spec_name] = artists
    
    return result_artists

def plot_phase_slope_vs_frequency(pattern, theta: float = 0.0, phi: float = 0.0,
                                 component: str = 'e_co',
                                 ax: Optional[plt.Axes] = None,
                                 fig_size: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot phase and derived quantities versus frequency for a specific point in the pattern.
    
    This shows the phase progression and calculated electrical length and group delay
    across the frequency band.
    
    Args:
        pattern: AntennaPattern object
        theta: Theta angle in degrees (default: 0.0 - boresight)
        phi: Phi angle in degrees (default: 0.0)
        component: Field component to analyze ('e_co', 'e_cx', 'e_theta', 'e_phi')
        ax: Optional matplotlib axes to plot on
        fig_size: Figure size as (width, height) in inches
        
    Returns:
        matplotlib.Figure: The created figure object
    """
    if len(pattern.frequencies) < 2:
        raise ValueError("Pattern must have at least 2 frequencies")
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    
    # Find nearest angles
    theta_val, theta_idx = find_nearest(pattern.theta_angles, theta)
    phi_val, phi_idx = find_nearest(pattern.phi_angles, phi)
    
    frequencies = pattern.frequencies
    
    # Get unwrapped phase data at the specified point (all frequencies)
    # This returns phase in degrees
    phase_data_deg = pattern.get_phase(component, unwrapped=True)[:, theta_idx, phi_idx]
    
    # Calculate electrical length and group delay at valid frequencies (need derivatives)
    electrical_lengths = []
    group_delays = []
    valid_frequencies = []
    
    if len(frequencies) >= 3:
        # Convert phase to radians for derivative calculation
        phase_data_rad = np.radians(phase_data_deg)
        
        # Use simple 3-point central difference for interior points
        for i in range(1, len(frequencies) - 1):
            # Central difference: (f[i+1] - f[i-1]) / (2*h)
            df = frequencies[i+1] - frequencies[i-1]  # Total frequency span
            dphi = phase_data_rad[i+1] - phase_data_rad[i-1]  # Total phase change in radians
            
            # Phase slope in rad/Hz
            phase_slope = dphi / df
            
            # Electrical length in degrees at this frequency
            electrical_length_deg = -(phase_slope / (2 * np.pi))  * frequencies[i] * 360
            electrical_lengths.append(electrical_length_deg)
            
            # Group delay in nanoseconds
            # Formula: τ = -dφ/dω where φ is in radians, ω is in rad/s
            # Since ω = 2πf, we have dφ/dω = (dφ/df) / (2π)
            # Result is in seconds, convert to ns
            group_delay_ns = -(phase_slope / (2 * np.pi)) * 1e9
            group_delays.append(group_delay_ns)
            
            valid_frequencies.append(frequencies[i])
    
    # Create primary plot for phase
    line1 = ax.plot(frequencies / 1e6, phase_data_deg, 'b-o', markersize=4, 
                    label='Phase')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Phase (degrees)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    
    # Create secondary y-axis for electrical length (degrees)
    if electrical_lengths:
        ax2 = ax.twinx()
        line2 = ax2.plot(np.array(valid_frequencies) / 1e6, electrical_lengths, 'r-s', markersize=4,
                         label='Electrical Length')
        ax2.set_ylabel('Electrical Length (degrees)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    # Create third y-axis for group delay
    if group_delays:
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        line3 = ax3.plot(np.array(valid_frequencies) / 1e6, group_delays, 'g-^', markersize=4,
                         label='Group Delay')
        ax3.set_ylabel('Group Delay (ns)', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
    
    # Add title and legend
    ax.set_title(f'Phase vs Frequency at θ={theta_val:.1f}°, φ={phi_val:.1f}°\n'
                f'Component: {component}')
    
    # Combine legends
    lines = line1
    if electrical_lengths:
        lines += line2
    if group_delays:
        lines += line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    fig.tight_layout()
    return fig

def plot_pattern_2d_polar(
    pattern: AntennaPattern,
    frequency: Optional[float] = None,
    component: str = 'e_co',
    value_type: Literal['gain', 'phase', 'axial_ratio'] = 'gain',
    unwrap_phase: bool = True,
    normalize: bool = False,  # Add this
    ax: Optional[plt.Axes] = None,
    fig_size: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis'
) -> Tuple[plt.Figure, Optional[Any]]:
    """
    Create a 2D color plot of the antenna pattern using polar projection.
    
    Automatically detects if pattern data is in 'central' format (common for antenna
    measurements) and temporarily converts to 'sided' format for proper 2D polar 
    visualization covering the full sphere.
    
    In the polar projection:
    - Theta becomes the radial coordinate (distance from center)  
    - Phi becomes the angular coordinate
    - Boresight (theta=0°) is at the center
    
    Args:
        pattern: AntennaPattern object
        frequency: Specific frequency to plot in Hz, or None to use first frequency
        component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')
        value_type: Type of value to plot ('gain', 'phase', or 'axial_ratio')
        unwrap_phase: If True and value_type is 'phase', unwrap phase discontinuities
        ax: Optional matplotlib axes with polar projection, or None to create new
        fig_size: Figure size as (width, height) in inches  
        title: Optional title for the plot
        colorbar: If True, add colorbar to the plot
        vmin: Minimum value for color scale (auto if None)
        vmax: Maximum value for color scale (auto if None)
        cmap: Colormap name for the plot
        
    Returns:
        Tuple[matplotlib.Figure, colorbar]: The created figure and colorbar objects
        
    Raises:
        ValueError: If component or value_type is invalid
    """
    # Validate inputs
    valid_components = ['e_co', 'e_cx', 'e_theta', 'e_phi']
    if component not in valid_components:
        raise ValueError(f"Invalid component: {component}. Must be one of {valid_components}")
    
    valid_value_types = ['gain', 'phase', 'axial_ratio']  
    if value_type not in valid_value_types:
        raise ValueError(f"Invalid value_type: {value_type}. Must be one of {valid_value_types}")
    
    # Special case: axial ratio only works with co/cx components
    if value_type == 'axial_ratio' and component not in ['e_co', 'e_cx']:
        raise ValueError("Axial ratio plotting requires component to be 'e_co' or 'e_cx'")
    
    # Create a working copy of the pattern to avoid modifying the original
    plot_pattern = pattern.copy()
    
    # Detect coordinate format and convert to sided for 2D polar plot
    theta_angles = plot_pattern.theta_angles
    phi_angles = plot_pattern.phi_angles
    
    theta_min, theta_max = np.min(theta_angles), np.max(theta_angles)
    phi_min, phi_max = np.min(phi_angles), np.max(phi_angles)
    
    # Check if data is in central format (common for antenna measurements)
    is_central = (theta_min < 0 or phi_max < 300)  # Simple heuristic for central data
    is_sided = (theta_min >= 0 and theta_max <= 180 and phi_min >= 0 and phi_max > 300)
    
    print(f"Pattern coordinate format detected:")
    print(f"  Theta range: {theta_min:.1f}° to {theta_max:.1f}°")
    print(f"  Phi range: {phi_min:.1f}° to {phi_max:.1f}°")
    
    # Convert to sided format for proper 2D visualization if needed
    if not is_sided:
        print("  Converting to sided format (0-180°, 0-360°) for 2D polar plot")
        try:
            plot_pattern.transform_coordinates('sided')
            theta_angles = plot_pattern.theta_angles
            phi_angles = plot_pattern.phi_angles
            print(f"  Converted - Theta: {np.min(theta_angles):.1f}° to {np.max(theta_angles):.1f}°")
            print(f"  Converted - Phi: {np.min(phi_angles):.1f}° to {np.max(phi_angles):.1f}°")
        except Exception as e:
            print(f"  Warning: Could not convert coordinates: {e}")
            print("  Plotting with original coordinates")
    
    # Handle frequency selection
    frequencies = plot_pattern.frequencies
    if frequency is None:
        freq_idx = 0
        selected_frequency = frequencies[0]
    else:
        selected_frequency, freq_idx = find_nearest(frequencies, frequency)
    
    # Create coordinate meshgrids
    phi_rad = np.deg2rad(phi_angles)  # Convert phi to radians for polar plot
    theta_mesh, phi_mesh = np.meshgrid(theta_angles, phi_rad, indexing='ij')
    
    # Extract data based on value_type and component
    if value_type == 'axial_ratio':
        # Get axial ratio data [freq, theta, phi] 
        data_3d = plot_pattern.get_axial_ratio()
        plot_data = data_3d[freq_idx, :, :]
        units = 'dB'
        default_title = f'Axial Ratio at {selected_frequency/1e9:.2f} GHz'
    elif value_type == 'gain':
        # Get gain data in dB
        data_3d = plot_pattern.get_gain_db(component)
        plot_data = data_3d[freq_idx, :, :]
        units = 'dB'
        comp_label = component.replace('_', '-').upper()
        default_title = f'{comp_label} Gain at {selected_frequency/1e9:.2f} GHz'
    elif value_type == 'phase':
        # Get phase data in degrees
        data_3d = plot_pattern.get_phase(component, unwrapped=unwrap_phase)
        plot_data = data_3d[freq_idx, :, :]
        units = 'degrees'
        comp_label = component.replace('_', '-').upper()
        default_title = f'{comp_label} Phase at {selected_frequency/1e9:.2f} GHz'

    # apply normalization if requested 
    if normalize and value_type == 'gain':
        peak_value = np.max(plot_data)
        plot_data = plot_data - peak_value
    elif normalize and value_type == 'phase':
        ref_phase = plot_data[0, 0]  # First theta, first phi point
        plot_data = plot_data - ref_phase
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(projection='polar'))
    else:
        fig = ax.figure
        if ax.name != 'polar':
            raise ValueError("Provided axes must have polar projection")
    
    # Create the polar color plot
    # Note: pcolormesh expects (phi, theta) order for polar coordinates
    im = ax.pcolormesh(phi_mesh.T, theta_mesh.T, plot_data.T, 
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    # Configure polar plot
    ax.set_theta_zero_location('N')  # Put phi=0 at top
    ax.set_theta_direction(-1)       # Clockwise phi direction (standard antenna convention)
    
    # Set radial axis label and limits
    ax.set_ylabel('Theta (degrees)', labelpad=30)
    ax.set_ylim(0, np.max(theta_angles))
    
    # Set angular axis ticks (phi in degrees)
    ax.set_thetagrids(np.arange(0, 360, 45))
    
    # Add title
    if title is None:
        title = default_title
    ax.set_title(title, pad=20)
    
    # Add colorbar
    cbar = None
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label(f'{value_type.capitalize()} ({units})')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, cbar

def detect_coordinate_format(pattern: AntennaPattern) -> str:
    """
    Detect whether pattern data is in 'central' or 'sided' coordinate format.
    
    Args:
        pattern: AntennaPattern object
        
    Returns:
        str: 'central', 'sided', or 'unknown'
    """
    theta_angles = pattern.theta_angles
    phi_angles = pattern.phi_angles
    
    theta_min, theta_max = np.min(theta_angles), np.max(theta_angles)
    phi_min, phi_max = np.min(phi_angles), np.max(phi_angles)
    
    # Check for sided format: theta 0:180, phi 0:360
    is_sided = (theta_min >= -5 and theta_max <= 185 and 
                phi_min >= -5 and phi_max >= 355)
    
    # Check for central format: theta around ±180, phi 0:180
    is_central = (theta_min < -5 and theta_max > 175 and
                  phi_min >= -5 and phi_max <= 185)
    
    # Alternative central: theta 0:180, phi limited range
    is_central_alt = (theta_min >= -5 and theta_max <= 185 and
                      phi_min >= -5 and phi_max <= 185)
    
    if is_sided:
        return 'sided'
    elif is_central or is_central_alt:
        return 'central'
    else:
        return 'unknown'