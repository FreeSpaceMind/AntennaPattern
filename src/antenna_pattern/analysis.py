"""
Analysis functions for antenna radiation patterns.
"""
import numpy as np
import logging
from scipy import optimize
from typing import Dict, Tuple, Optional, List, Union
import xarray as xr

from .utilities import find_nearest, frequency_to_wavelength
from .pattern_functions import unwrap_phase, phase_pattern_translate
from .polarization import polarization_tp2rl

# Configure logging
logger = logging.getLogger(__name__)


def calculate_phase_center(pattern, theta_angle: float, frequency: Optional[float] = None, 
                       n_iter: int = 10) -> np.ndarray:
    """
    Finds the optimum phase center given a theta angle and frequency.
    
    The optimum phase center is the point that, when used as the origin,
    minimizes the phase variation across the beam within the +/- theta_angle range.
    Uses basinhopping optimization to find the global minimum.
    
    Args:
        pattern: AntennaPattern object
        theta_angle: Angle in degrees to optimize phase center for
        frequency: Optional specific frequency to use, or None to use first frequency
        n_iter: Number of iterations for basinhopping
        
    Returns:
        np.ndarray: [x, y, z] coordinates of the optimum phase center
    """
        
    # Get data arrays
    freq_array = pattern.data.frequency.values
    theta_array = pattern.data.theta.values
    phi_array = pattern.data.phi.values
    
    # Handle frequency selection
    if frequency is None:
        freq_idx = 0
        freq = freq_array[freq_idx]
    else:
        freq_val, freq_idx = find_nearest(freq_array, frequency)
        if isinstance(freq_idx, np.ndarray):
            freq_idx = freq_idx.item()
        freq = freq_array[freq_idx]
    
    # Find the indices corresponding to +/- theta_angle
    theta_n, idx_n = find_nearest(theta_array, -theta_angle)
    theta_p, idx_p = find_nearest(theta_array, theta_angle)
    
    if isinstance(idx_n, np.ndarray):
        idx_n = idx_n.item()
    if isinstance(idx_p, np.ndarray):
        idx_p = idx_p.item()
    
    # Ensure idx_n < idx_p (in case theta_array isn't sorted)
    if idx_n > idx_p:
        idx_n, idx_p = idx_p, idx_n
    
    # Get original co-pol data
    e_co = pattern.data.e_co.values[freq_idx, :, :]
    
    # Define cost function for optimization
    def phase_center_cost(translation):
        """Calculate phase spread/flatness after applying translation."""
        # Convert theta and phi to radians for phase_pattern_translate
        theta_rad = np.radians(theta_array)
        phi_rad = np.radians(phi_array)
        
        # Get original phase
        co_phase = np.angle(e_co)
        
        # Apply the translation to the phase pattern
        translated_phase = phase_pattern_translate(
            freq, theta_rad, phi_rad, translation, co_phase)
        
        # Extract the region of interest (+/- theta_angle)
        roi_phase = translated_phase[idx_n:idx_p+1, :]
        
        # Initialize array to hold unwrapped phase for each phi cut
        unwrapped_phases = np.zeros_like(roi_phase)
        
        # Unwrap phase along theta for each phi cut
        for phi_idx in range(roi_phase.shape[1]):
            # Extract the 1D array for this phi cut
            phi_cut = roi_phase[:, phi_idx]
            # Unwrap it - since it's now 1D, axis=0 is correct
            unwrapped_phi_cut = unwrap_phase(phi_cut, discont=np.pi)
            # Store the result
            unwrapped_phases[:, phi_idx] = unwrapped_phi_cut
        
        # Calculate flatness metric (overall deviation from zero across whole region)
        theta_indices = np.arange(unwrapped_phases.shape[0])
        boresight_idx = np.argmin(np.abs(theta_indices - (len(theta_indices) // 2)))
        boresight_phase = np.mean(unwrapped_phases[boresight_idx, :])
        normalized_phases = unwrapped_phases - boresight_phase
        flatness_metric = np.std(normalized_phases)
        
        # Return appropriate metric based on method
        return flatness_metric
    
    # Calculate wavelength for scaled step sizes
    wavelength = 3e8 / freq  # Speed of light / frequency
    step_size = wavelength / 20  # Reasonable step size based on wavelength
    
    # Define a step-taking function for basinhopping
    class PhaseStepTaker(object):
        def __init__(self, stepsize=step_size):
            self.stepsize = stepsize
            
        def __call__(self, x):
            # Take random steps proportional to wavelength
            x_new = np.copy(x)
            x_new += np.random.uniform(-self.stepsize, self.stepsize, x.shape)
            return x_new
    
    # Define a simple bounds function
    def bounds_check(x_new, **kwargs):
        """Check if the new position is within bounds."""
        max_value = 2.0  # Max distance in meters
        return bool(np.all(np.abs(x_new) < max_value))
        
    # Run basinhopping
    initial_guess = np.zeros(3)
    minimizer_kwargs = {"method": "Nelder-Mead"}
    
    result = optimize.basinhopping(
        phase_center_cost,
        initial_guess,
        niter=n_iter,
        T=1.0,  # Temperature parameter (higher allows escaping deeper local minima)
        stepsize=step_size,
        take_step=PhaseStepTaker(),
        accept_test=bounds_check,
        minimizer_kwargs=minimizer_kwargs
    )
    
    translation = result.x
    
    # Check for reasonable results
    if np.any(np.isnan(translation)) or np.any(np.isinf(translation)):
        logger.warning("Optimization returned invalid values, using zeros")
        translation = np.zeros(3)
        
    # Limit to reasonable range
    max_value = 2  # 2 meters
    if np.any(np.abs(translation) > max_value):
        logger.warning(f"Limiting excessive translation values: {translation}")
        translation = np.clip(translation, -max_value, max_value)
        
    return translation

def principal_plane_phase_center(frequency, theta1, theta2, theta3, phase1, phase2, phase3):
    """
    Calculate phase center using three points on a principal plane.
    
    Args:
        frequency: Frequency in Hz
        theta1: Theta angle (radians) of first point
        theta2: Theta angle (radians) of second point
        theta3: Theta angle (radians) of third point
        phase1: Phase (radians) of first point
        phase2: Phase (radians) of second point
        phase3: Phase (radians) of third point
        
    Returns:
        Tuple[ndarray, ndarray]: Planar and z-axis displacement
    """
    
    wavelength = frequency_to_wavelength(frequency)
    wavenumber = 2 * np.pi / wavelength
    
    # Ensure all inputs are arrays
    if np.isscalar(theta1): theta1 = np.array([theta1])
    if np.isscalar(theta2): theta2 = np.array([theta2])
    if np.isscalar(theta3): theta3 = np.array([theta3])
    if np.isscalar(phase1): phase1 = np.array([phase1])
    if np.isscalar(phase2): phase2 = np.array([phase2])
    if np.isscalar(phase3): phase3 = np.array([phase3])
    
    # Compute denominators first to check for division by zero
    denom1 = ((np.cos(theta2) - np.cos(theta3)) * (np.sin(theta2) - np.sin(theta1))) - \
            ((np.cos(theta2) - np.cos(theta1)) * (np.sin(theta2) - np.sin(theta3)))
    
    # Avoid division by zero
    if np.any(np.abs(denom1) < 1e-10):
        logger.warning("Small denominator detected in phase center calculation")
        denom1 = np.where(np.abs(denom1) < 1e-10, 1e-10 * np.sign(denom1), denom1)
    
    planar_displacement = (1 / wavenumber) * (
        (
            ((phase2 - phase1) * (np.cos(theta2) - np.cos(theta3)))
            - ((phase2 - phase3) * (np.cos(theta2) - np.cos(theta1)))
        ) / denom1
    )
    
    zaxis_displacement = (1 / wavenumber) * (
        (
            ((phase2 - phase3) * (np.sin(theta2) - np.sin(theta1)))
            - ((phase2 - phase1) * (np.sin(theta2) - np.sin(theta3)))
        ) / denom1
    )
        
    # Check for invalid results
    if np.any(np.isnan(planar_displacement)) or np.any(np.isinf(planar_displacement)) or \
        np.any(np.isnan(zaxis_displacement)) or np.any(np.isinf(zaxis_displacement)):
        logger.warning("Phase center calculation produced invalid values")
        return np.zeros_like(planar_displacement), np.zeros_like(zaxis_displacement)
    
    return planar_displacement.flatten(), zaxis_displacement.flatten()


def get_axial_ratio(pattern):
    """
    Calculate the axial ratio (ratio of major to minor axis of polarization ellipse).
    
    Args:
        pattern: AntennaPattern object
        
    Returns:
        xr.DataArray: Axial ratio (linear scale)
    """

    # Convert to circular polarization components if not already
    if pattern.polarization in ['rhcp', 'lhcp']:
        e_r = pattern.data.e_co if pattern.polarization == 'rhcp' else pattern.data.e_cx
        e_l = pattern.data.e_cx if pattern.polarization == 'rhcp' else pattern.data.e_co
    else:
        # Need to calculate circular components
        e_r, e_l = polarization_tp2rl(
            pattern.data.phi.values,
            pattern.data.e_theta.values, 
            pattern.data.e_phi.values
        )
        e_r = xr.DataArray(
            e_r, 
            dims=pattern.data.e_theta.dims,
            coords=pattern.data.e_theta.coords
        )
        e_l = xr.DataArray(
            e_l, 
            dims=pattern.data.e_theta.dims,
            coords=pattern.data.e_theta.coords
        )
    
    # Calculate axial ratio
    er_mag = np.abs(e_r)
    el_mag = np.abs(e_l)
    
    # Handle pure circular polarization case
    min_val = 1e-15
    er_mag = xr.where(er_mag < min_val, min_val, er_mag)
    el_mag = xr.where(el_mag < min_val, min_val, el_mag)
    
    # Calculate axial ratio
    return 20 * np.log10((er_mag + el_mag) / np.maximum(np.abs(er_mag - el_mag), min_val))

"""
Phase length and group delay analysis functions for antenna patterns.
Add these functions to analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
import xarray as xr
from scipy import interpolate

from .utilities import find_nearest, lightspeed

def get_phase_length(pattern, frequency: float, theta: float = 0.0, phi: float = 0.0, 
                    component: str = 'e_co') -> float:
    """
    Calculate the electrical phase length of the antenna at a specific point.
    
    This function computes the phase slope with respect to frequency and converts
    it to an electrical length representing the total path length from the input
    port to the specified observation point.
    
    Args:
        pattern: AntennaPattern object
        frequency: Frequency in Hz at which to calculate phase length
        theta: Theta angle in degrees (default: 0.0 - boresight)
        phi: Phi angle in degrees (default: 0.0)
        component: Field component to analyze ('e_co', 'e_cx', 'e_theta', 'e_phi')
        
    Returns:
        float: Electrical phase length in meters
        
    Raises:
        ValueError: If pattern doesn't have multiple frequencies for slope calculation
        ValueError: If specified angles are outside pattern coverage
    """
    frequencies = pattern.frequencies
    
    if len(frequencies) < 2:
        raise ValueError("Pattern must have at least 2 frequencies to calculate phase slope")
    
    # Find nearest angles
    theta_val, theta_idx = find_nearest(pattern.theta_angles, theta)
    phi_val, phi_idx = find_nearest(pattern.phi_angles, phi)
    
    # Get unwrapped phase data at the specified point
    phase_data = pattern.get_phase(component, unwrapped=True)[:, theta_idx, phi_idx]
    
    # Convert phase to radians
    phase_rad = np.radians(phase_data)
    
    # Calculate phase slope using linear regression around the specified frequency
    # Use a window of frequencies around the target frequency for better accuracy
    freq_idx = np.argmin(np.abs(frequencies - frequency))
    
    # Define window size (use all frequencies if less than 5, otherwise use 5-point window)
    window_size = min(5, len(frequencies))
    half_window = window_size // 2
    
    # Define frequency window indices
    start_idx = max(0, freq_idx - half_window)
    end_idx = min(len(frequencies), freq_idx + half_window + 1)
    
    freq_window = frequencies[start_idx:end_idx]
    phase_window = phase_rad[start_idx:end_idx]
    
    # Calculate phase slope (dφ/df) using linear regression
    coeffs = np.polyfit(freq_window, phase_window, 1)
    phase_slope = coeffs[0]  # radians/Hz
    
    # Convert to electrical length: L = (dφ/df) / (2π) * c
    electrical_length = (phase_slope / (2 * np.pi)) * lightspeed
    
    return electrical_length

def get_group_delay(pattern, frequency: float, theta: float = 0.0, phi: float = 0.0,
                   component: str = 'e_co') -> float:
    """
    Calculate the group delay of the antenna at a specific point.
    
    Group delay is the negative derivative of phase with respect to frequency:
    τ_group = -dφ/df
    
    Args:
        pattern: AntennaPattern object
        frequency: Frequency in Hz at which to calculate group delay
        theta: Theta angle in degrees (default: 0.0 - boresight)
        phi: Phi angle in degrees (default: 0.0)
        component: Field component to analyze ('e_co', 'e_cx', 'e_theta', 'e_phi')
        
    Returns:
        float: Group delay in seconds
        
    Raises:
        ValueError: If pattern doesn't have multiple frequencies for slope calculation
        ValueError: If specified angles are outside pattern coverage
    """
    frequencies = pattern.frequencies
    
    if len(frequencies) < 2:
        raise ValueError("Pattern must have at least 2 frequencies to calculate group delay")
    
    # Find nearest angles
    theta_val, theta_idx = find_nearest(pattern.theta_angles, theta)
    phi_val, phi_idx = find_nearest(pattern.phi_angles, phi)
    
    # Get unwrapped phase data at the specified point
    phase_data = pattern.get_phase(component, unwrapped=True)[:, theta_idx, phi_idx]
    
    # Convert phase to radians
    phase_rad = np.radians(phase_data)
    
    # Calculate phase slope using same windowing approach as phase_length
    freq_idx = np.argmin(np.abs(frequencies - frequency))
    
    window_size = min(5, len(frequencies))
    half_window = window_size // 2
    
    start_idx = max(0, freq_idx - half_window)
    end_idx = min(len(frequencies), freq_idx + half_window + 1)
    
    freq_window = frequencies[start_idx:end_idx]
    phase_window = phase_rad[start_idx:end_idx]
    
    # Calculate phase slope (dφ/df) using linear regression
    coeffs = np.polyfit(freq_window, phase_window, 1)
    phase_slope = coeffs[0]  # radians/Hz
    
    # Group delay is negative phase slope
    group_delay = -phase_slope  # seconds
    
    return group_delay