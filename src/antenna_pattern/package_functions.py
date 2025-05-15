"""
Functions for working across multiple antenna patterns.
"""
import numpy as np
from typing import List, Optional, Union, Dict, Any

from .pattern import AntennaPattern
from .pattern_functions import unwrap_phase

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
    
    This function computes the difference between two patterns by subtracting
    amplitude in dB and phase separately. The patterns must have compatible 
    dimensions (same theta, phi, and frequency values).
    
    Args:
        pattern1: First AntennaPattern object
        pattern2: Second AntennaPattern object
            
    Returns:
        AntennaPattern: A new antenna pattern containing the difference (pattern1 - pattern2)
        
    Raises:
        ValueError: If patterns have incompatible dimensions
    """
    import numpy as np
    
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
    
    # Get the field components
    e_theta1 = pattern1.data.e_theta.values
    e_phi1 = pattern1.data.e_phi.values
    
    e_theta2 = pattern2.data.e_theta.values
    e_phi2 = pattern2.data.e_phi.values
    
    # Calculate amplitude in dB
    amp_theta1 = 20 * np.log10(np.abs(e_theta1) + 1e-15)  # Avoid log(0)
    amp_phi1 = 20 * np.log10(np.abs(e_phi1) + 1e-15)
    
    amp_theta2 = 20 * np.log10(np.abs(e_theta2) + 1e-15)
    amp_phi2 = 20 * np.log10(np.abs(e_phi2) + 1e-15)
    
    # Calculate phases (in radians)
    phase_theta1 = np.angle(e_theta1)
    phase_phi1 = np.angle(e_phi1)
    
    phase_theta2 = np.angle(e_theta2)
    phase_phi2 = np.angle(e_phi2)
    
    # Compute differences
    amp_theta_diff = amp_theta1 - amp_theta2
    amp_phi_diff = amp_phi1 - amp_phi2
    
    # For phase, handle circular difference to get shortest path
    phase_theta_diff = unwrap_phase(phase_theta1) - unwrap_phase(phase_theta2)
    phase_phi_diff = unwrap_phase(phase_phi1) - unwrap_phase(phase_phi2)
    
    # Convert amplitude differences back to linear scale
    amp_theta_linear = 10**(amp_theta_diff / 20)
    amp_phi_linear = 10**(amp_phi_diff / 20)
    
    # Reconstruct complex values with difference amplitude and phase
    e_theta_diff = amp_theta_linear * np.exp(1j * phase_theta_diff)
    e_phi_diff = amp_phi_linear * np.exp(1j * phase_phi_diff)
    
    # Create metadata for the difference pattern
    metadata = {
        'source': 'difference_pattern',
        'pattern1_polarization': pattern1.polarization,
        'pattern2_polarization': pattern2.polarization,
        'operations': []
    }
    
    # Include original pattern metadata if available
    if hasattr(pattern1, 'metadata') and pattern1.metadata:
        metadata['pattern1_metadata'] = pattern1.metadata
    if hasattr(pattern2, 'metadata') and pattern2.metadata:
        metadata['pattern2_metadata'] = pattern2.metadata
    
    # Create a new pattern with the difference data
    return AntennaPattern(
        theta=theta1,
        phi=phi1,
        frequency=freq1,
        e_theta=e_theta_diff,
        e_phi=e_phi_diff,
        metadata=metadata
    )