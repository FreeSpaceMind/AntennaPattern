"""
Basic tests for the AntennaPattern package.
"""
import os
import sys

# Add the src directory to the path if not already installed
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pytest
from antenna_pattern import AntennaPattern, polarization_tp2xy, polarization_tp2rl


def test_create_antenna_pattern():
    """Test creating a simple AntennaPattern object."""
    # Create test data
    theta = np.linspace(-180, 180, 10)
    phi = np.array([0, 90])
    frequency = np.array([10e9])  # 10 GHz
    
    # Create simple field patterns
    e_theta = np.zeros((len(frequency), len(theta), len(phi)), dtype=complex)
    e_phi = np.zeros((len(frequency), len(theta), len(phi)), dtype=complex)
    
    # Fill with a simple cosine pattern
    for t_idx, t_val in enumerate(theta):
        for p_idx, p_val in enumerate(phi):
            e_theta[0, t_idx, p_idx] = np.cos(np.radians(t_val/2))
    
    # Create antenna pattern
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization="theta"
    )
    
    # Verify basic properties
    assert pattern.polarization == "theta"
    assert len(pattern.frequencies) == 1
    assert len(pattern.theta_angles) == 10
    assert len(pattern.phi_angles) == 2
    
    # Verify data was correctly stored
    np.testing.assert_allclose(pattern.data.e_theta.values, e_theta)
    np.testing.assert_allclose(pattern.data.e_phi.values, e_phi)


def test_polarization_conversion():
    """Test basic polarization conversion functions."""
    # Create test data
    phi = np.array([0, 45, 90])
    e_theta = np.array([1+0j, 0.7+0.7j, 0+1j])
    e_phi = np.array([0+0j, 0.7-0.7j, 1+0j])
    
    # Test tp2xy conversion
    e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
    
    # For phi=0, e_x should equal e_theta and e_y should equal -e_phi
    np.testing.assert_allclose(e_x[0], e_theta[0])
    np.testing.assert_allclose(e_y[0], -e_phi[0])
    
    # Test tp2rl conversion
    e_r, e_l = polarization_tp2rl(phi, e_theta, e_phi)
    
    # Simple verification (could be expanded)
    assert e_r.shape == e_theta.shape
    assert e_l.shape == e_phi.shape


def test_change_polarization():
    """Test changing polarization of an antenna pattern."""
    # Create simple pattern
    theta = np.linspace(-90, 90, 7)
    phi = np.array([0, 90])
    frequency = np.array([10e9])
    e_theta = np.ones((1, len(theta), len(phi)), dtype=complex)
    e_phi = np.zeros((1, len(theta), len(phi)), dtype=complex)
    
    # Create pattern with theta polarization
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization="theta"
    )
    
    # Change to RHCP
    rhcp_pattern = pattern.change_polarization("rhcp")
    
    # Verify the change
    assert rhcp_pattern.polarization == "rhcp"
    assert pattern.polarization == "theta"  # Original unchanged
    
    
def test_scale_pattern_scalar():
    """Test scaling pattern amplitude with a scalar dB value."""
    # Create simple pattern
    theta = np.linspace(-90, 90, 7)
    phi = np.array([0, 90])
    frequency = np.array([10e9])
    e_theta = np.ones((1, len(theta), len(phi)), dtype=complex)
    e_phi = np.zeros((1, len(theta), len(phi)), dtype=complex)
    
    # Create pattern
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization="theta"
    )
    
    # Scale by 3 dB
    scaled_pattern = pattern.scale_pattern(3.0)
    
    # Expected amplitude should be multiplied by 10^(3/20) = 1.4125
    expected_amplitude = 1.4125
    
    # Check scaling
    np.testing.assert_allclose(np.abs(scaled_pattern.data.e_theta.values), 
                               expected_amplitude * np.abs(pattern.data.e_theta.values),
                               rtol=1e-4)
    np.testing.assert_allclose(np.abs(scaled_pattern.data.e_phi.values), 
                               expected_amplitude * np.abs(pattern.data.e_phi.values),
                               rtol=1e-4)
    
    # Original pattern should be unchanged
    np.testing.assert_allclose(np.abs(pattern.data.e_theta.values), 1.0)


def test_scale_pattern_frequency_vector():
    """Test scaling pattern amplitude with a frequency-dependent vector."""
    # Create pattern with multiple frequencies
    theta = np.linspace(-90, 90, 7)
    phi = np.array([0, 90])
    frequency = np.array([2e9, 4e9, 6e9])  # 3 frequencies
    e_theta = np.ones((3, len(theta), len(phi)), dtype=complex)
    e_phi = np.zeros((3, len(theta), len(phi)), dtype=complex)
    
    # Create pattern
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization="theta"
    )
    
    # Case 1: Matching frequency vector
    scale_db = np.array([2.0, 3.0, 4.0])  # Different scale for each frequency
    scaled_pattern = pattern.scale_pattern(scale_db)
    
    # Expected amplitude factors
    expected_factors = 10**(scale_db/20.0).reshape(-1, 1, 1)
    
    # Check scaling
    np.testing.assert_allclose(np.abs(scaled_pattern.data.e_theta.values), 
                               expected_factors * np.abs(pattern.data.e_theta.values),
                               rtol=1e-4)
    
    # Case 2: Custom frequency vector needing interpolation
    custom_freq = np.array([1e9, 5e9, 7e9])
    custom_scale = np.array([1.0, 5.0, 7.0])
    
    scaled_pattern2 = pattern.scale_pattern(custom_scale, freq_scale=custom_freq)
    
    # Should interpolate to approximately [2.0, 4.0, 6.0] dB at our frequencies
    interp_scale = np.array([2.0, 4.0, 6.0])
    expected_factors2 = 10**(interp_scale/20.0).reshape(-1, 1, 1)
    
    # Check interpolated scaling
    np.testing.assert_allclose(np.abs(scaled_pattern2.data.e_theta.values), 
                               expected_factors2 * np.abs(pattern.data.e_theta.values),
                               rtol=1e-1)  # Less strict tolerance for interpolation


def test_scale_pattern_2d():
    """Test scaling pattern amplitude with a 2D array for freq/phi combinations."""
    # Create pattern with multiple frequencies and phi angles
    theta = np.linspace(-90, 90, 5)
    phi = np.array([0, 45, 90])
    frequency = np.array([2e9, 4e9])
    e_theta = np.ones((2, len(theta), len(phi)), dtype=complex)
    e_phi = np.zeros((2, len(theta), len(phi)), dtype=complex)
    
    # Create pattern
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization="theta"
    )
    
    # Create 2D scaling array [freq, phi]
    scale_2d = np.array([
        [1.0, 2.0, 3.0],  # Values for first frequency
        [4.0, 5.0, 6.0]   # Values for second frequency
    ])
    
    # Scale with matching freq/phi arrays
    scaled_pattern = pattern.scale_pattern(scale_2d, freq_scale=frequency, phi_scale=phi)
    
    # Convert scaling to amplitude factors and reshape for broadcasting
    scale_factors = 10**(scale_2d/20.0).reshape(2, 1, 3)
    
    # Check scaling at each freq/phi combination
    for f_idx in range(len(frequency)):
        for p_idx in range(len(phi)):
            expected_amp = scale_factors[f_idx, 0, p_idx]
            actual_amp = np.abs(scaled_pattern.data.e_theta.values[f_idx, :, p_idx])
            np.testing.assert_allclose(actual_amp, expected_amp, rtol=1e-4)
            
    # Test with custom freq/phi grids needing interpolation
    custom_freq = np.array([1e9, 5e9])
    custom_phi = np.array([0, 60, 120])
    custom_scale = np.array([
        [0.0, 1.5, 3.0],
        [3.0, 4.5, 6.0]
    ])
    
    scaled_pattern2 = pattern.scale_pattern(custom_scale, 
                                           freq_scale=custom_freq, 
                                           phi_scale=custom_phi)
    
    # This requires 2D interpolation - just check a few key points
    # First freq, first phi angle - should be interpolated to ~0.5 dB
    expected_factor = 10**(0.5/20.0)
    actual_factor = np.abs(scaled_pattern2.data.e_theta.values[0, 0, 0])
    np.testing.assert_allclose(actual_factor, expected_factor, rtol=1e-1)
    
    # Second freq, last phi angle - should be interpolated to ~5.5 dB
    expected_factor = 10**(5.5/20.0)
    actual_factor = np.abs(scaled_pattern2.data.e_theta.values[1, 0, 2])
    np.testing.assert_allclose(actual_factor, expected_factor, rtol=1e-1)
    
    # Check co-pol/cross-pol components
    # For pure theta polarization converted to RHCP:
    # Both RHCP and LHCP components should be present with equal magnitude
    mag_co = np.abs(rhcp_pattern.data.e_co.values)
    mag_cx = np.abs(rhcp_pattern.data.e_cx.values)
    np.testing.assert_allclose(mag_co, mag_cx)


if __name__ == "__main__":
    # Run tests manually
    test_create_antenna_pattern()
    test_polarization_conversion()
    test_change_polarization()
    print("All tests passed!")