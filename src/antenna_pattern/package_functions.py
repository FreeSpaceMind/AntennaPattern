"""
Functions for working across multiple antenna patterns.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any

from .pattern import AntennaPattern

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
    
    This function converts both patterns to a common coordinate system (Ludwig's III),
    computes the complex difference directly, and then converts back. This preserves
    the physical meaning of the subtraction operation.
    
    Args:
        pattern1: First AntennaPattern object
        pattern2: Second AntennaPattern object
            
    Returns:
        AntennaPattern: A new antenna pattern containing the difference
    """
    import numpy as np
    from .polarization import polarization_tp2xy, polarization_xy2tp
    
    # Verify compatible dimensions
    theta1 = pattern1.theta_angles
    phi1 = pattern1.phi_angles
    freq1 = pattern1.frequencies
    
    if not (np.array_equal(theta1, pattern2.theta_angles) and 
            np.array_equal(phi1, pattern2.phi_angles) and 
            np.array_equal(freq1, pattern2.frequencies)):
        raise ValueError("Patterns have incompatible dimensions")
    
    # Get field components
    e_theta1 = pattern1.data.e_theta.values
    e_phi1 = pattern1.data.e_phi.values
    e_theta2 = pattern2.data.e_theta.values
    e_phi2 = pattern2.data.e_phi.values
    
    # Initialize result arrays
    e_theta_diff = np.zeros_like(e_theta1, dtype=complex)
    e_phi_diff = np.zeros_like(e_phi1, dtype=complex)
    
    # Process each frequency
    for f_idx in range(len(freq1)):
        # Convert both patterns to Ludwig's III
        e_x1, e_y1 = polarization_tp2xy(phi1, e_theta1[f_idx], e_phi1[f_idx])
        e_x2, e_y2 = polarization_tp2xy(phi1, e_theta2[f_idx], e_phi2[f_idx])
        
        # Instead of division, take the logarithmic difference (for amplitudes in dB)
        # and phase difference directly
        
        # Amplitude calculation in dB domain
        amp_x1 = 20 * np.log10(np.abs(e_x1) + 1e-15)
        amp_y1 = 20 * np.log10(np.abs(e_y1) + 1e-15)
        amp_x2 = 20 * np.log10(np.abs(e_x2) + 1e-15)
        amp_y2 = 20 * np.log10(np.abs(e_y2) + 1e-15)
        
        # Amplitude differences in dB
        amp_x_diff = amp_x1 - amp_x2
        amp_y_diff = amp_y1 - amp_y2
        
        # Phase calculations (unwrapped to handle jumps)
        phase_x1 = np.angle(e_x1)
        phase_y1 = np.angle(e_y1)
        phase_x2 = np.angle(e_x2)
        phase_y2 = np.angle(e_y2)
        
        # Calculate phase differences (with proper wrapping)
        phase_x_diff = np.mod(phase_x1 - phase_x2 + np.pi, 2*np.pi) - np.pi
        phase_y_diff = np.mod(phase_y1 - phase_y2 + np.pi, 2*np.pi) - np.pi
        
        # Reconstruct complex field values from amplitude and phase
        amp_x_linear = 10**(amp_x_diff / 20)
        amp_y_linear = 10**(amp_y_diff / 20)
        
        e_x_diff = amp_x_linear * np.exp(1j * phase_x_diff)
        e_y_diff = amp_y_linear * np.exp(1j * phase_y_diff)
        
        # Convert back to spherical coordinates
        e_theta_diff[f_idx], e_phi_diff[f_idx] = polarization_xy2tp(phi1, e_x_diff, e_y_diff)
    
    # Create result pattern
    result_pattern = AntennaPattern(
        theta=theta1,
        phi=phi1,
        frequency=freq1,
        e_theta=e_theta_diff,
        e_phi=e_phi_diff,
        polarization=pattern1.polarization,
        metadata={
            'source': 'difference_pattern',
            'method': 'amplitude_phase_difference',
            'operations': []
        }
    )
    
    return result_pattern