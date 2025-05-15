"""
Functions for working across multiple antenna patterns.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any

from .pattern import AntennaPattern
from .polarization import polarization_tp2xy, polarization_xy2tp, polarization_rl2tp

def average_patterns(patterns: List[AntennaPattern], weights: Optional[List[float]] = None) -> AntennaPattern:
    """
    Create a new antenna pattern by averaging multiple patterns.
    
    This function computes a weighted average of the provided patterns. All patterns
    must have compatible dimensions (same theta, phi, and frequency values).
    
    Args:
        patterns: List of AntennaPattern objects to average
        weights: Optional list of weights for each pattern. If None, equal weights are used.
            Weights will be normalized to sum to 1.
            
    Returns:
        AntennaPattern: A new antenna pattern containing the weighted average
        
    Raises:
        ValueError: If patterns have incompatible dimensions
        ValueError: If weights are provided but don't match the number of patterns
    """
    if len(patterns) < 1:
        raise ValueError("At least one pattern is required for averaging")
    
    # Get reference dimensions from first pattern
    theta = patterns[0].theta_angles
    phi = patterns[0].phi_angles
    freq = patterns[0].frequencies
    
    # Check that all patterns have the same dimensions
    for i, pattern in enumerate(patterns[1:], 1):
        if not np.array_equal(pattern.theta_angles, theta):
            raise ValueError(f"Pattern {i} has different theta angles than pattern 0")
        if not np.array_equal(pattern.phi_angles, phi):
            raise ValueError(f"Pattern {i} has different phi angles than pattern 0")
        if not np.array_equal(pattern.frequencies, freq):
            raise ValueError(f"Pattern {i} has different frequencies than pattern 0")
    
    # Handle weights
    if weights is None:
        # Equal weights
        weights = np.ones(len(patterns)) / len(patterns)
    else:
        if len(weights) != len(patterns):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of patterns ({len(patterns)})")
        
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
    
    # Initialize output arrays
    e_theta_avg = np.zeros((len(freq), len(theta), len(phi)), dtype=complex)
    e_phi_avg = np.zeros((len(freq), len(theta), len(phi)), dtype=complex)
    
    # Compute weighted average
    for i, pattern in enumerate(patterns):
        e_theta_avg += weights[i] * pattern.data.e_theta.values
        e_phi_avg += weights[i] * pattern.data.e_phi.values
    
    # Create combined polarization string and metadata
    polarizations = [pattern.polarization for pattern in patterns]
    
    # Create metadata for the averaged pattern
    metadata = {
        'source': 'averaged_pattern',
        'weights': weights.tolist(),
        'source_polarizations': polarizations,
        'operations': []
    }
    
    # Include original pattern metadata if available
    for i, pattern in enumerate(patterns):
        if hasattr(pattern, 'metadata') and pattern.metadata:
            metadata[f'source_pattern_{i}_metadata'] = pattern.metadata
    
    # Create a new pattern with the averaged data
    return AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=freq,
        e_theta=e_theta_avg,
        e_phi=e_phi_avg,
        metadata=metadata
    )

def difference_patterns(
    pattern1: AntennaPattern, 
    pattern2: AntennaPattern
) -> AntennaPattern:
    """
    Create a new antenna pattern representing the difference between two patterns.
    Works directly with co-polarized component of the first pattern.
    
    Args:
        pattern1: First AntennaPattern object (typically the original pattern)
        pattern2: Second AntennaPattern object (typically the processed pattern)
            
    Returns:
        AntennaPattern: A new antenna pattern containing the difference (pattern1/pattern2)
                        with polarization matching pattern1
    """
    # Check that patterns have the same dimensions
    theta1 = pattern1.theta_angles
    phi1 = pattern1.phi_angles
    freq1 = pattern1.frequencies
    
    theta2 = pattern2.theta_angles
    phi2 = pattern2.phi_angles
    freq2 = pattern2.frequencies
    
    # Verify the patterns have compatible dimensions
    if not np.array_equal(theta1, theta2):
        raise ValueError("Patterns have different theta angles")
    if not np.array_equal(phi1, phi2):
        raise ValueError("Patterns have different phi angles")
    if not np.array_equal(freq1, freq2):
        raise ValueError("Patterns have different frequencies")
    
    # Find boresight index (closest to theta=0)
    boresight_idx = np.argmin(np.abs(theta1))
    
    # Ensure both patterns have the same polarization
    pol = pattern1.polarization
    if pattern2.polarization != pol:
        pattern2 = pattern2.change_polarization(pol)
    
    # Get the field components - work directly with co-pol and cross-pol
    e_co1 = pattern1.data.e_co.values
    e_cx1 = pattern1.data.e_cx.values
    
    e_co2 = pattern2.data.e_co.values
    e_cx2 = pattern2.data.e_cx.values
    
    # Initialize arrays for difference pattern
    e_co_diff = np.zeros_like(e_co1, dtype=complex)
    e_cx_diff = np.zeros_like(e_cx1, dtype=complex)
    
    # Process each frequency separately
    for f_idx in range(len(freq1)):
        # Avoid division by zero
        epsilon = 1e-15
        safe_e_co2 = np.where(np.abs(e_co2[f_idx]) < epsilon, epsilon, e_co2[f_idx])
        safe_e_cx2 = np.where(np.abs(e_cx2[f_idx]) < epsilon, epsilon, e_cx2[f_idx])
        
        # Compute ratios directly in co/cross coordinate system
        e_co_diff[f_idx] = e_co1[f_idx] / safe_e_co2
        e_cx_diff[f_idx] = e_cx1[f_idx] / safe_e_cx2
        
        # Normalize phase based on first phi cut at boresight
        reference_phase = np.angle(e_co_diff[f_idx, boresight_idx, 0])
        phase_correction = np.exp(-1j * reference_phase)
        
        e_co_diff[f_idx] *= phase_correction
        e_cx_diff[f_idx] *= phase_correction
    
    # Convert back to e_theta and e_phi based on polarization
    e_theta_diff = np.zeros_like(e_co_diff, dtype=complex)
    e_phi_diff = np.zeros_like(e_cx_diff, dtype=complex)
    
    if pol in ('rhcp', 'rh', 'r'):
        # For RHCP: e_co = e_r, e_cx = e_l
        # Convert from RHCP/LHCP to theta/phi
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_rl2tp(phi1, e_co_diff[f_idx], e_cx_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    elif pol in ('lhcp', 'lh', 'l'):
        # For LHCP: e_co = e_l, e_cx = e_r
        # Convert from LHCP/RHCP to theta/phi
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_rl2tp(phi1, e_cx_diff[f_idx], e_co_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    elif pol in ('x', 'l3x'):
        # For X: e_co = e_x, e_cx = e_y
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_xy2tp(phi1, e_co_diff[f_idx], e_cx_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    elif pol in ('y', 'l3y'):
        # For Y: e_co = e_y, e_cx = e_x
        for f_idx in range(len(freq1)):
            e_theta_temp, e_phi_temp = polarization_xy2tp(phi1, e_cx_diff[f_idx], e_co_diff[f_idx])
            e_theta_diff[f_idx] = e_theta_temp
            e_phi_diff[f_idx] = e_phi_temp
    else:
        # For theta/phi, directly use values
        if pol == 'theta':
            e_theta_diff = e_co_diff
            e_phi_diff = e_cx_diff
        else:  # phi polarization
            e_theta_diff = e_cx_diff
            e_phi_diff = e_co_diff
    
    # Create metadata for the difference pattern
    metadata = {
        'source': 'difference_pattern',
        'pattern1_polarization': pattern1.polarization,
        'pattern2_polarization': pattern2.polarization,
        'difference_method': 'direct_co_cx_ratio',
        'operations': []
    }
    
    # Create a new pattern with the difference data
    result_pattern = AntennaPattern(
        theta=theta1,
        phi=phi1,
        frequency=freq1,
        e_theta=e_theta_diff,
        e_phi=e_phi_diff,
        polarization=pattern1.polarization,
        metadata=metadata
    )
    
    return result_pattern