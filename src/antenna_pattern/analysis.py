"""
Analysis functions for antenna radiation patterns.
"""
import numpy as np
import logging
from scipy import optimize
from typing import Dict, Tuple, Optional, List, Union
import xarray as xr

from .utilities import find_nearest, unwrap_phase, frequency_to_wavelength
from .polarization import phase_pattern_translate, polarization_tp2rl
from .utilities import lightspeed, interpolate_crossing

# Configure logging
logger = logging.getLogger(__name__)


def find_phase_center(pattern, theta_angle: float, frequency: Optional[float] = None) -> np.ndarray:
    """
    Finds the optimum phase center given a theta angle and frequency.
    
    The optimum phase center is the point that, when used as the origin,
    minimizes the phase variation across the beam from -theta_angle to +theta_angle.
    
    Args:
        pattern: AntennaPattern object
        theta_angle: Angle in degrees to optimize phase center for
        frequency: Optional specific frequency to use, or None to use all frequencies
    
    Returns:
        np.ndarray: [x, y, z] coordinates of the optimum phase center
    """
    # Validate theta_angle
    if (theta_angle < 0 or theta_angle > np.max(pattern.theta_angles)):
        logger.warning(f"Theta angle {theta_angle} is outside the available range [0, {np.max(pattern.theta_angles)}]")
        theta_angle = min(max(0, theta_angle), np.max(pattern.theta_angles))
        
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
    
    # Get unwrapped phase data for co-polarization
    co_pol_phase = np.angle(pattern.data.e_co.values[freq_idx, :, :])
    co_pol_phase_unwrap = unwrap_phase(co_pol_phase, discont=np.pi)
    
    # Define a cost function for optimization
    def phase_spread_cost(translation):
        """Calculate phase spread after applying translation."""
        # Convert theta and phi to radians for phase_pattern_translate
        theta_rad = np.radians(theta_array)
        phi_rad = np.radians(phi_array)
        
        # Apply the translation to the phase pattern
        translated_phase = phase_pattern_translate(
            freq, theta_rad, phi_rad, translation, co_pol_phase_unwrap)
        
        # Extract the region of interest (+/- theta_angle)
        roi = translated_phase[idx_n:idx_p+1, :]
        
        # Unwrap the phase again AFTER translation
        unwrapped_roi = np.zeros_like(roi)
        for phi_idx in range(roi.shape[1]):
            unwrapped_roi[:, phi_idx] = unwrap_phase(roi[:, phi_idx], discont=np.pi)
        
        # Calculate the total phase spread across all phi cuts
        phase_min = np.min(unwrapped_roi)
        phase_max = np.max(unwrapped_roi)
        spread = phase_max - phase_min
        
        return spread
    
    # Use Nelder-Mead optimizer
    initial_guess = np.zeros(3)
    
    result = optimize.minimize(
        phase_spread_cost, 
        initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    if result.success:
        translation = result.x
    else:
        logger.warning(f"Optimization did not converge: {result.message}")
        translation = result.x  # Still use the best result found
    
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

def beamwidth_from_pattern(gain_pattern: np.ndarray, angles: np.ndarray, level_db: float = -3.0) -> float:
    """
    Calculate beamwidth at specified level from a gain pattern.
    
    Args:
        gain_pattern: Gain pattern in dB
        angles: Corresponding angle values in degrees
        level_db: Level relative to maximum at which to measure beamwidth (default: -3 dB)
    
    Returns:
        Beamwidth in degrees
        
    Raises:
        ValueError: If pattern or angles array is empty
        ValueError: If no points at or below the specified level are found
    """
    if len(gain_pattern) == 0 or len(angles) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if len(gain_pattern) != len(angles):
        raise ValueError(f"Length mismatch: gain_pattern ({len(gain_pattern)}) != angles ({len(angles)})")
    
    # Find the maximum gain value and its index
    max_gain = np.max(gain_pattern)
    threshold = max_gain + level_db
    
    # Find indices where the gain crosses the threshold
    above_threshold = gain_pattern >= threshold
    
    # Find the transition points (where the boolean array changes from True to False or vice versa)
    transitions = np.where(np.diff(above_threshold))[0]
    
    if len(transitions) < 2:
        raise ValueError(f"Could not find two crossing points at {level_db} dB level")
    
    # For multiple crossings, use the widest crossing points around the maximum
    max_idx = np.argmax(gain_pattern)
    
    # Find crossing points before and after the maximum
    left_crossings = transitions[transitions < max_idx]
    right_crossings = transitions[transitions > max_idx]
    
    if len(left_crossings) == 0 or len(right_crossings) == 0:
        raise ValueError(f"Maximum is too close to the edge of the pattern - could not find crossings on both sides")
    
    left_idx = left_crossings[-1]
    right_idx = right_crossings[0]
    
    # Interpolate to find the exact angles where the pattern crosses the threshold
    left_angle = interpolate_crossing(angles[left_idx:left_idx+2], gain_pattern[left_idx:left_idx+2], threshold)
    right_angle = interpolate_crossing(angles[right_idx:right_idx+2], gain_pattern[right_idx:right_idx+2], threshold)
    
    # Calculate beamwidth
    beamwidth = abs(right_angle - left_angle)
    
    return beamwidth

def calculate_beamwidth(pattern, frequency: Optional[float] = None, level_db: float = -3.0) -> Dict[str, float]:
    """
    Calculate the beamwidth at specified level for principal planes.
    
    Args:
        pattern: AntennaPattern object
        frequency: Optional frequency to calculate beamwidth for, or None for all
        level_db: Level relative to maximum at which to measure beamwidth (default: -3 dB)
        
    Returns:
        Dict[str, float]: Beamwidths in degrees for E and H planes
    """
    # Handle frequency selection
    if frequency is not None:
        with pattern.at_frequency(frequency) as single_freq_pattern:
            return calculate_beamwidth(single_freq_pattern, level_db=level_db)
    
    # Get the gain in dB
    gain_db = pattern.get_gain_db('e_co')
    
    # Find the maximum gain and its indices
    max_gain = gain_db.max()
    max_indices = np.unravel_index(np.argmax(gain_db.values), gain_db.shape)
    max_freq_idx, max_theta_idx, max_phi_idx = max_indices
    
    # Get closest cardinal plane indices
    phi_0_idx = find_nearest(pattern.phi_angles, 0)[1]
    phi_90_idx = find_nearest(pattern.phi_angles, 90)[1]
    
    # E-plane cut (phi = 0° or closest)
    e_plane_cut = gain_db[max_freq_idx, :, phi_0_idx].values
    
    # H-plane cut (phi = 90° or closest)
    h_plane_cut = gain_db[max_freq_idx, :, phi_90_idx].values
    
    # Calculate beamwidths using utility function
    e_plane_bw = beamwidth_from_pattern(e_plane_cut, pattern.theta_angles, level_db)
    h_plane_bw = beamwidth_from_pattern(h_plane_cut, pattern.theta_angles, level_db)
    
    return {
        'E_plane': e_plane_bw,
        'H_plane': h_plane_bw,
        'Average': (e_plane_bw + h_plane_bw) / 2
    }

def apply_mars(pattern, maximum_radial_extent: float):
    """
    Apply Mathematical Absorber Reflection Suppression technique.
    
    The MARS algorithm transforms antenna measurement data to mitigate reflections
    from the measurement chamber. It is particularly effective for electrically
    large antennas.
    
    Args:
        pattern: AntennaPattern object
        maximum_radial_extent: Maximum radial extent of the antenna in meters
        
    Returns:
        New AntennaPattern with MARS algorithm applied
    """
    
    if maximum_radial_extent <= 0:
        raise ValueError("Maximum radial extent must be positive")
    
    frequency = pattern.data.frequency.values
    theta = pattern.data.theta.values
    phi = pattern.data.phi.values
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    # Initialize outputs
    e_theta_new = np.empty_like(e_theta)
    e_phi_new = np.empty_like(e_phi)
    
    # Apply MARS algorithm
    for f_idx, f in enumerate(frequency):
        # Calculate wavenumber and coefficients range
        wavenumber = 2 * np.pi * f / lightspeed
        max_coefficients = int(np.floor(wavenumber * maximum_radial_extent))
        coefficients = np.arange(-max_coefficients, max_coefficients + 1, 1)
        
        # Create arrays for theta in radians
        theta_rad = np.radians(theta)
        
        # Initialize storage arrays for cylindrical coefficients
        CMC_1_sum = np.zeros_like(e_theta[f_idx, :, :], dtype=complex)
        CMC_2_sum = np.zeros_like(e_phi[f_idx, :, :], dtype=complex)
        
        # Precompute exponential terms for efficiency
        exp_terms = np.zeros((len(coefficients), len(theta)), dtype=complex)
        for n_idx, n in enumerate(coefficients):
            exp_terms[n_idx, :] = np.exp(-1j * n * theta_rad)
        
        # Process each coefficient
        for n_idx, n in enumerate(coefficients):
            # Compute mode coefficient for theta component
            CMC_1 = (
                -1 * ((-1j) ** (-n)) / (4 * np.pi * wavenumber) *
                np.trapz(
                    (e_theta[f_idx, :, :].transpose() * exp_terms[n_idx, :]).transpose(),
                    theta_rad, axis=0
                )
            )
            
            # Compute mode coefficient for phi component
            CMC_2 = (
                -1j * ((-1j) ** (-n)) / (4 * np.pi * wavenumber) *
                np.trapz(
                    (e_phi[f_idx, :, :].transpose() * exp_terms[n_idx, :]).transpose(),
                    theta_rad, axis=0
                )
            )
            
            # Sum the modes
            CMC_1_term = np.outer(exp_terms[n_idx, :], (-1j) ** n * CMC_1)
            CMC_2_term = np.outer(exp_terms[n_idx, :], (-1j) ** n * CMC_2)
            
            CMC_1_sum += CMC_1_term
            CMC_2_sum += CMC_2_term
        
        # Compute final field components
        e_phi_new[f_idx, :, :] = 2 * 1j * wavenumber * CMC_2_sum
        e_theta_new[f_idx, :, :] = -2 * wavenumber * CMC_1_sum
    
    # Flip the theta axis because of coordinate system difference from reference
    e_theta_flipped = np.flip(e_theta_new, axis=1)
    e_phi_flipped = np.flip(e_phi_new, axis=1)
    
    # Import here to avoid circular import
    from .pattern import AntennaPattern
    
    return AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta_flipped,
        e_phi=e_phi_flipped,
        polarization=pattern.polarization
    )


def translate_phase_pattern(pattern, translation, normalize=True):
    """
    Shifts the antenna phase pattern to place the origin at the location defined by the shift.
    
    Args:
        pattern: AntennaPattern object to translate
        translation: [x, y, z] translation vector in meters
        normalize: if true, normalize the translated phase pattern to zero degrees
        
    Returns:
        AntennaPattern: New pattern with translated phase
    """
    
    # Get underlying numpy arrays
    frequency = pattern.data.frequency.values
    theta = pattern.data.theta.values
    phi = pattern.data.phi.values
    e_theta = pattern.data.e_theta.values
    e_phi = pattern.data.e_phi.values
    
    # Get amplitude and phase components
    e_theta_phase = np.angle(e_theta)
    e_phi_phase = np.angle(e_phi)
    e_theta_mag = np.abs(e_theta)
    e_phi_mag = np.abs(e_phi)
    
    # Convert angles to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    # Apply translation to phase patterns
    e_theta_phase_new = phase_pattern_translate(
        frequency, theta_rad, phi_rad, translation, e_theta_phase
    )
    e_phi_phase_new = phase_pattern_translate(
        frequency, theta_rad, phi_rad, translation, e_phi_phase
    )
    
    # Reconstruct complex values
    e_theta_new = e_theta_mag * np.exp(1j * e_theta_phase_new)
    e_phi_new = e_phi_mag * np.exp(1j * e_phi_phase_new)

    # normalize
    if normalize:
        # Find the indices for theta=0, phi=0 (or closest values)
        theta_0_idx = np.argmin(np.abs(theta))
        phi_0_idx = np.argmin(np.abs(phi))
        
        # Convert e_theta and e_phi to e_x and e_y at the reference point
        from .polarization import polarization_tp2xy
        
        # Get the phase reference using Ludwig-3 components
        for f_idx in range(len(frequency)):
            # Convert theta/phi to x/y for this frequency
            e_x_single, e_y_single = polarization_tp2xy(
                phi, 
                e_theta_new[f_idx], 
                e_phi_new[f_idx]
            )
            
            # Get the phase at reference point for e_y
            ref_phase = np.angle(e_y_single[theta_0_idx, phi_0_idx])
            
            # Apply phase normalization by subtracting reference phase
            # This preserves relative phase relationships
            phase_correction = np.exp(-1j * ref_phase)
            e_theta_new[f_idx] = e_theta_new[f_idx] * phase_correction
            e_phi_new[f_idx] = e_phi_new[f_idx] * phase_correction
    
    # Import here to avoid circular import
    from .pattern import AntennaPattern
    
    # Create new antenna pattern
    return AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta_new,
        e_phi=e_phi_new,
        polarization=pattern.polarization
    )


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
    return (er_mag + el_mag) / np.maximum(np.abs(er_mag - el_mag), min_val)