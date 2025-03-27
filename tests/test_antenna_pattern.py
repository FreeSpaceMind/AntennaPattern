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