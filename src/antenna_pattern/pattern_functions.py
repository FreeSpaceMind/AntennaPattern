"""
Functions that perform operations on AntennaPattern objects.
These functions contained here to shorten the AntennaPattern definition file (pattern.py)
"""

import numpy as np
from typing import Tuple, Union, Optional, List, Any, Callable
from scipy.interpolate import interp1d

from .utilities import lightspeed, frequency_to_wavelength
from .polarization import polarization_tp2xy, polarization_tp2rl, polarization_xy2pt

def change_polarization(pattern_obj, new_polarization: str) -> None:
    """
    Change the polarization of the antenna pattern.
    
    Args:
        pattern_obj: AntennaPattern object to modify
        new_polarization: New polarization type to use
        
    Raises:
        ValueError: If the new polarization is invalid
    """
    # Simply call assign_polarization with the new polarization type
    pattern_obj.assign_polarization(new_polarization)
    
    # Clear cache due to change in polarization
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        pattern_obj.metadata['polarization'] = pattern_obj.polarization
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'change_polarization',
            'new_polarization': new_polarization
        })

def translate(pattern_obj, translation: np.ndarray, normalize: bool = True) -> None:
    """
    Shifts the antenna phase pattern to place the origin at the location defined by the shift.
    
    Args:
        pattern_obj: AntennaPattern object to modify
        translation: [x, y, z] translation vector in meters, can be 1D (applied to
            all frequencies) or 2D with shape (num_frequencies, 3)
        normalize: if true, normalize the translated phase pattern to zero degrees
    """
    
    # Get underlying numpy arrays
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values
    e_phi = pattern_obj.data.e_phi.values
    
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
    
    # Update the pattern data directly
    pattern_obj.data['e_theta'].values = e_theta_new
    pattern_obj.data['e_phi'].values = e_phi_new
    
    # Recalculate derived values
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Normalize if requested
    if normalize:
        normalize_phase(pattern_obj)
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'translate',
            'translation': translation.tolist() if isinstance(translation, np.ndarray) else translation,
            'normalize': normalize
        })

def normalize_phase(pattern_obj, reference_theta=0, reference_phi=0) -> None:
    """
    Normalize the phase of an antenna pattern based on its polarization type.
    
    This function sets the phase of the co-polarized component at the reference
    point (closest to reference_theta, reference_phi) to zero, while preserving
    the relative phase between components.
    
    Args:
        pattern_obj: AntennaPattern object to modify
        reference_theta: Reference theta angle in degrees (default: 0)
        reference_phi: Reference phi angle in degrees (default: 0)
    """
    
    # Get underlying numpy arrays
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values
    e_phi = pattern_obj.data.e_phi.values
    
    # Find the indices for reference angles (or closest values)
    theta_ref_idx = np.argmin(np.abs(theta - reference_theta))
    phi_ref_idx = np.argmin(np.abs(phi - reference_phi))
    
    # Get reference angle values (for logging)
    theta_ref_actual = theta[theta_ref_idx]
    phi_ref_actual = phi[phi_ref_idx]
    
    # Determine which component to use as reference based on polarization
    pol = pattern_obj.polarization.lower()
    
    # Process each frequency separately
    for f_idx in range(len(frequency)):
        # Select reference component based on polarization type
        if pol in ('theta', 'phi'):
            # For spherical polarization, use the corresponding component
            if pol == 'theta':
                ref_phase = np.angle(e_theta[f_idx, theta_ref_idx, phi_ref_idx])
            else:  # phi polarization
                ref_phase = np.angle(e_phi[f_idx, theta_ref_idx, phi_ref_idx])
        
        elif pol in ('x', 'l3x', 'y', 'l3y'):
            # For Ludwig-3 polarization, calculate e_x and e_y
            e_x, e_y = polarization_tp2xy(
                phi, 
                e_theta[f_idx], 
                e_phi[f_idx]
            )
            if pol in ('x', 'l3x'):
                ref_phase = np.angle(e_x[theta_ref_idx, phi_ref_idx])
            else:  # y polarization
                ref_phase = np.angle(e_y[theta_ref_idx, phi_ref_idx])
        
        elif pol in ('rhcp', 'rh', 'r', 'lhcp', 'lh', 'l'):
            # For circular polarization, calculate RHCP and LHCP components
            e_r, e_l = polarization_tp2rl(
                phi,
                e_theta[f_idx],
                e_phi[f_idx]
            )
            if pol in ('rhcp', 'rh', 'r'):
                ref_phase = np.angle(e_r[theta_ref_idx, phi_ref_idx])
            else:  # LHCP polarization
                ref_phase = np.angle(e_l[theta_ref_idx, phi_ref_idx])
        
        else:
            # Fallback to e_theta for unknown polarization
            ref_phase = np.angle(e_theta[f_idx, theta_ref_idx, phi_ref_idx])
        
        # Apply phase normalization by subtracting reference phase
        # This preserves relative phase relationships
        phase_correction = np.exp(-1j * ref_phase)
        e_theta[f_idx] = e_theta[f_idx] * phase_correction
        e_phi[f_idx] = e_phi[f_idx] * phase_correction
    
    # Update the pattern data directly
    pattern_obj.data['e_theta'].values = e_theta
    pattern_obj.data['e_phi'].values = e_phi
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'normalize_phase',
            'reference_theta': reference_theta,
            'reference_phi': reference_phi,
            'actual_theta': float(theta_ref_actual),
            'actual_phi': float(phi_ref_actual)
        })


def apply_mars(pattern_obj, maximum_radial_extent: float) -> None:
    """
    Apply Mathematical Absorber Reflection Suppression technique.
    
    Args:
        pattern_obj: AntennaPattern object to modify
        maximum_radial_extent: Maximum radial extent of the antenna in meters
    """
    
    if maximum_radial_extent <= 0:
        raise ValueError("Maximum radial extent must be positive")
    
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values.copy()
    e_phi = pattern_obj.data.e_phi.values.copy()
    
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
    
    # Update the pattern data directly
    pattern_obj.data['e_theta'].values = e_theta_flipped
    pattern_obj.data['e_phi'].values = e_phi_flipped
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'apply_mars',
            'maximum_radial_extent': maximum_radial_extent
        })

def swap_polarization_axes(pattern_obj) -> None:
    """
    Swap vertical and horizontal polarization ports.
    
    Args:
        pattern_obj: AntennaPattern object to modify
    """
    
    # Get data
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values 
    e_phi = pattern_obj.data.e_phi.values
    
    # Convert to x/y and back to swap the axes
    e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
    e_theta_new, e_phi_new = polarization_xy2pt(phi, e_y, e_x)  # Note: x and y are swapped
    
    # Update the pattern data directly
    pattern_obj.data['e_theta'].values = e_theta_new
    pattern_obj.data['e_phi'].values = e_phi_new
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'swap_polarization_axes'
        })

def scale_pattern(pattern_obj, scale_db: Union[float, np.ndarray], 
                  freq_scale: Optional[np.ndarray] = None,
                  phi_scale: Optional[np.ndarray] = None) -> None:
    """
    Scale the amplitude of the antenna pattern by the input value in dB.
    
    This method supports several input formats:
    1. A scalar value: Apply the same scaling to all frequencies and angles
    2. A 1D array matching frequency length: Apply frequency-dependent scaling
    3. A 1D array with custom frequencies: Interpolate to pattern frequencies
    4. A 2D array with scaling values for freq/phi combinations
    
    Args:
        pattern_obj: AntennaPattern object to modify
        scale_db: Scaling values in dB. Can be:
            - float: Single value applied to all frequencies and angles
            - 1D array[freq]: Values for each frequency in pattern
            - 2D array[freq, phi]: Values for each frequency/phi combination
        freq_scale: Optional frequency vector in Hz when scale_db doesn't match
            pattern frequency points. Required if scale_db is 1D and doesn't 
            match pattern frequencies, or if scale_db is 2D.
        phi_scale: Optional phi angle vector in degrees when scale_db is 2D and 
            doesn't match pattern phi points.
            
    Raises:
        ValueError: If input arrays have incompatible dimensions
    """
    
    # Get pattern dimensions
    pattern_freq = pattern_obj.frequencies
    pattern_phi = pattern_obj.phi_angles
    pattern_theta = pattern_obj.theta_angles
    n_freq = len(pattern_freq)
    n_phi = len(pattern_phi)
    n_theta = len(pattern_theta)
    
    # Get field components
    e_theta = pattern_obj.data.e_theta.values.copy()
    e_phi = pattern_obj.data.e_phi.values.copy()
    
    # Case 1: Single scalar value - apply uniformly
    if np.isscalar(scale_db):
        # Apply scaling to field components
        pattern_obj.data['e_theta'].values = scale_amplitude(e_theta, scale_db)
        pattern_obj.data['e_phi'].values = scale_amplitude(e_phi, scale_db)
        
        # Update metadata
        if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
            if 'operations' not in pattern_obj.metadata:
                pattern_obj.metadata['operations'] = []
            pattern_obj.metadata['operations'].append({
                'type': 'scale_pattern',
                'scale_db': float(scale_db)
            })
        
        # Recalculate derived components and clear cache
        pattern_obj.assign_polarization(pattern_obj.polarization)
        pattern_obj.clear_cache()
        return
    
    # Convert to numpy array if not already
    scale_db = np.asarray(scale_db)
    
    # Case 2: 1D array matching frequency length
    if scale_db.ndim == 1 and len(scale_db) == n_freq and freq_scale is None:
        # Reshape for broadcasting - add axes for theta and phi dimensions
        scale_factor = scale_db.reshape(-1, 1, 1)
        
        # Apply scaling to field components
        pattern_obj.data['e_theta'].values = scale_amplitude(e_theta, scale_factor)
        pattern_obj.data['e_phi'].values = scale_amplitude(e_phi, scale_factor)
        
        # Update metadata
        if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
            if 'operations' not in pattern_obj.metadata:
                pattern_obj.metadata['operations'] = []
            pattern_obj.metadata['operations'].append({
                'type': 'scale_pattern',
                'scale_db': scale_db.tolist()
            })
        
        # Recalculate derived components and clear cache
        pattern_obj.assign_polarization(pattern_obj.polarization)
        pattern_obj.clear_cache()
        return
    
    # Case 3: 1D array with custom frequencies - need interpolation
    if scale_db.ndim == 1 and freq_scale is not None:
        if len(scale_db) != len(freq_scale):
            raise ValueError(f"scale_db length ({len(scale_db)}) must match freq_scale length ({len(freq_scale)})")
        
        # Interpolate to match pattern frequencies
        interp_func = interp1d(freq_scale, scale_db, bounds_error=False, fill_value="extrapolate")
        interp_scale = interp_func(pattern_freq)
        
        # Set reasonable limits to prevent overflow
        interp_scale = np.clip(interp_scale, -50.0, 50.0)
        
        # Reshape for broadcasting
        scale_factor = interp_scale.reshape(-1, 1, 1)
        
        # Apply scaling to field components
        pattern_obj.data['e_theta'].values = scale_amplitude(e_theta, scale_factor)
        pattern_obj.data['e_phi'].values = scale_amplitude(e_phi, scale_factor)
        
        # Update metadata
        if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
            if 'operations' not in pattern_obj.metadata:
                pattern_obj.metadata['operations'] = []
            pattern_obj.metadata['operations'].append({
                'type': 'scale_pattern',
                'scale_db': scale_db.tolist(),
                'freq_scale': freq_scale.tolist(),
                'interpolated_scale': interp_scale.tolist()
            })
        
        # Recalculate derived components and clear cache
        pattern_obj.assign_polarization(pattern_obj.polarization)
        pattern_obj.clear_cache()
        return
    
    # Case 4: 2D array - need interpolation for both frequency and phi
    if scale_db.ndim == 2:
        if freq_scale is None:
            raise ValueError("freq_scale must be provided when scale_db is 2D")
        
        # Handle phi_scale
        if phi_scale is None:
            if scale_db.shape[1] != n_phi:
                raise ValueError(f"scale_db phi dimension ({scale_db.shape[1]}) "
                                f"must match pattern phi count ({n_phi})")
            phi_scale = pattern_phi
        
        if len(freq_scale) != scale_db.shape[0] or len(phi_scale) != scale_db.shape[1]:
            raise ValueError(f"scale_db shape ({scale_db.shape}) must match "
                            f"(freq_scale length, phi_scale length) = ({len(freq_scale)}, {len(phi_scale)})")
        
        # Set reasonable limits for dB values to prevent overflow
        max_db_value = 50.0
        min_db_value = -50.0
        
        # Create a 3D array to store scaling factors for each point in the pattern
        scale_factors = np.zeros((n_freq, n_theta, n_phi))
        
        # Use a simple nearest-neighbor approach for each (frequency, phi) point
        # This avoids any complex interpolation that could distort your pattern
        for f_idx, freq in enumerate(pattern_freq):
            # Find nearest frequency in freq_scale
            f_nearest_idx = np.abs(freq_scale - freq).argmin()
            
            for p_idx, phi in enumerate(pattern_phi):
                # Find nearest phi in phi_scale, accounting for periodicity
                # Calculate the angular distance considering wrap-around
                phi_dists = np.abs(np.mod(phi_scale - phi + 180, 360) - 180)
                p_nearest_idx = np.argmin(phi_dists)
                
                # Get scaling value from the nearest point
                scale_val = scale_db[f_nearest_idx, p_nearest_idx]
                
                # Clip to reasonable range
                scale_val = np.clip(scale_val, min_db_value, max_db_value)
                
                # Assign to all theta values for this phi angle and frequency
                scale_factors[f_idx, :, p_idx] = scale_val
        
        # Convert dB to linear scale factors
        linear_scale_factors = 10**(scale_factors / 20.0)
        
        # Apply scaling using numpy broadcasting
        pattern_obj.data['e_theta'].values = e_theta * linear_scale_factors
        pattern_obj.data['e_phi'].values = e_phi * linear_scale_factors
        
        # Update metadata
        if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
            if 'operations' not in pattern_obj.metadata:
                pattern_obj.metadata['operations'] = []
            pattern_obj.metadata['operations'].append({
                'type': 'scale_pattern',
                'scale_db': '2D array (see dimensions)',
                'scale_dimensions': scale_db.shape,
                'freq_scale': freq_scale.tolist() if len(freq_scale) < 100 else f"length={len(freq_scale)}",
                'phi_scale': phi_scale.tolist() if len(phi_scale) < 100 else f"length={len(phi_scale)}"
            })
        
        # Recalculate derived components and clear cache
        pattern_obj.assign_polarization(pattern_obj.polarization)
        pattern_obj.clear_cache()
        return
    
    raise ValueError("Invalid scale_db format. Must be scalar, 1D or 2D array.")

def phase_pattern_translate(frequency, theta, phi, translation, phase_pattern):
    """
    Shifts the antenna phase pattern to place the origin at the location defined by the shift.
    
    Args:
        frequency: Frequency in Hz (scalar or array)
        theta: Array of theta angles in radians
        phi: Array of phi angles in radians
        translation: [x, y, z] translation vector in meters
        phase_pattern: Array of antenna phase pattern in radians
        
    Returns:
        ndarray: Shifted phase pattern
    """
    
    wavelength = frequency_to_wavelength(frequency)
    wavenumber = 2 * np.pi / wavelength
    
    # Handle single frequency case - reshape arrays if needed
    if np.isscalar(frequency) and phase_pattern.ndim == 2:
        # This is a single frequency case - no frequency dimension
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

        sin_theta = np.sin(theta_grid)
        cos_theta = np.cos(theta_grid)
        cos_phi = np.cos(phi_grid)
        sin_phi = np.sin(phi_grid)
        
        # Calculate phase shift  
        phase_shift = wavenumber * (
            translation[0] * cos_phi * sin_theta +
            translation[1] * sin_phi * sin_theta +
            translation[2] * cos_theta
        )
        
        # Apply the shift
        shifted_pattern = phase_pattern - phase_shift
        
    else:
        # Multi-frequency implementation
        # Create a tiled array for matrix operations
        theta_array = np.moveaxis(
            np.tile(theta, (np.size(phi), np.size(frequency), 1)), -1, 0
        )
        phi_array = np.tile(phi, (np.size(theta), np.size(frequency), 1)).swapaxes(1, 2)
        
        # Handle per-frequency translation if needed
        if np.ndim(translation) == 1:
            translation = np.tile(translation, (np.size(frequency), 1))
        
        # Calculate phase shift
        shifted_pattern = np.moveaxis(phase_pattern, 0, -1) - wavenumber * (
            translation[:, 0] * np.cos(phi_array) * np.sin(theta_array) +
            translation[:, 1] * np.sin(phi_array) * np.sin(theta_array) +
            translation[:, 2] * np.cos(theta_array)
        )
        
        # Move axes back to original order
        shifted_pattern = np.moveaxis(shifted_pattern, -1, 0)
    
    # Normalize to -π to π range
    shifted_pattern = shifted_pattern % (2 * np.pi)
    shifted_pattern[shifted_pattern > np.pi] -= 2 * np.pi
    
    return shifted_pattern

def translate_phase_pattern(pattern_obj, translation, normalize=True) -> None:
    """
    Shifts the antenna phase pattern to place the origin at the location defined by the shift.
    
    Args:
        pattern_obj: AntennaPattern object to modify
        translation: [x, y, z] translation vector in meters
        normalize: if true, normalize the translated phase pattern to zero degrees
    """
    from .polarization import phase_pattern_translate
    
    # Get underlying numpy arrays
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values
    e_phi = pattern_obj.data.e_phi.values
    
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

    # Update the pattern data directly
    pattern_obj.data['e_theta'].values = e_theta_new
    pattern_obj.data['e_phi'].values = e_phi_new
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Normalize if requested
    if normalize:
        normalize_phase(pattern_obj)
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'translate_phase_pattern',
            'translation': translation.tolist() if isinstance(translation, np.ndarray) else translation,
            'normalize': normalize
        })

def unwrap_phase(phase: np.ndarray, discont: float = np.pi) -> np.ndarray:
    """
    phase unwrapping with adjustable discontinuity threshold.
    
    Args:
        phase: Array of phase values in radians
        discont: Size of the discontinuity for unwrapping (default: π)
    
    Returns:
        Unwrapped phase array
    """
    return np.unwrap(phase, discont=discont, axis=0)

def scale_amplitude(values: np.ndarray, scale_db: Union[float, np.ndarray]) -> np.ndarray:
    """
    Scale complex amplitude values by multiplying by a linear factor derived from dB.
    
    Args:
        values: Complex array of field values to scale
        scale_db: Value(s) in dB to scale by, can be scalar or array
        
    Returns:
        np.ndarray: Scaled complex values
        
    Note:
        This multiplies the amplitude by 10^(scale_db/20), maintaining the phase.
    """
    # Convert dB scaling to linear scale factor (amplitude, not power)
    scale_factor = 10**(scale_db / 20.0)
    
    # Apply scaling factor to complex values (preserves phase)
    return values * scale_factor

def transform_tp2uvw(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms from antenna pattern spherical coordinates to direction cosines.
    
    Coordinate system:
    - w = 1 along z-axis (boresight), w > 0 is front hemisphere, w < 0 is back
    - u = 1 along x-axis (theta=90, phi=0)
    - v = 1 along y-axis (theta=90, phi=90)
    
    Args:
        theta: Array of theta angles in degrees (0 to 180)
        phi: Array of phi angles in degrees (0 to 360)
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Direction cosines (u, v, w)
    """
    # Convert to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    
    u = sin_theta * np.cos(phi_rad)
    v = sin_theta * np.sin(phi_rad)
    w = cos_theta
    
    return u, v, w


def transform_uvw2tp(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms from direction cosines to antenna pattern spherical coordinates.
    
    Handles full spherical coverage with:
    - theta in [-180, 180] degrees
    - phi in [0, 360] degrees
    """
    # Normalize direction cosines
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    eps = 1e-10  # Small epsilon to avoid division by zero
    safe_mag = np.maximum(magnitude, eps)
    
    u_norm = u / safe_mag
    v_norm = v / safe_mag
    w_norm = w / safe_mag
    
    # Calculate theta angle (magnitude) from z-axis
    theta_mag = np.degrees(np.arccos(np.clip(w_norm, -1.0, 1.0)))
    
    # Calculate phi in [0, 360] range
    phi = np.degrees(np.arctan2(v_norm, u_norm))
    phi = np.mod(phi, 360)  # Ensure phi is in [0, 360]
    
    # For points with u < 0 and phi in [0, 180], map to negative theta
    # For points with u < 0 and phi in [180, 360], map to negative theta
    # This consistently applies negative theta to points in the "back" hemisphere
    if np.isscalar(u_norm):
        # Scalar case
        if u_norm < 0:
            theta = -theta_mag
        else:
            theta = theta_mag
    else:
        # Array case
        # Initialize theta as positive
        theta = np.copy(theta_mag)
        
        # For points in the back hemisphere (u < 0), use negative theta
        back_hemisphere = (u_norm < 0)
        theta[back_hemisphere] = -theta_mag[back_hemisphere]
    
    # Handle poles (where u and v are nearly zero)
    pole_threshold = 1e-6
    rho = np.sqrt(u_norm**2 + v_norm**2)
    
    if np.isscalar(rho):
        if rho < pole_threshold:
            phi = 0.0  # At poles, default phi to 0
    else:
        phi = np.where(rho < pole_threshold, 0.0, phi)
    
    return theta, phi

def isometric_rotation(u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                   az: float, el: float, roll: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs an isometric rotation of direction cosines.
    
    The rotation is applied in the following order:
    1. Roll (around z-axis)
    2. Elevation (around x-axis)
    3. Azimuth (around y-axis)
    
    Args:
        u, v, w: Direction cosines
        az: Azimuth angle in degrees
        el: Elevation angle in degrees
        roll: Roll angle in degrees
        
    Returns:
        Rotated direction cosines (u', v', w')
    """
    # Convert angles to radians
    az_rad = np.radians(az)
    el_rad = np.radians(el)
    roll_rad = np.radians(roll)
    
    # Handle scalar inputs
    scalar_input = np.isscalar(u) and np.isscalar(v) and np.isscalar(w)
    if scalar_input:
        u, v, w = np.array([u]), np.array([v]), np.array([w])
    
    # Remember original shape
    original_shape = np.shape(u)
    
    # Prepare coordinates as column vectors
    coords = np.vstack([u.flatten(), v.flatten(), w.flatten()])
    
    # Define rotation matrices (for right-handed coordinate system)
    # Roll - rotation around z-axis
    R_z = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    # Elevation - rotation around x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(el_rad), -np.sin(el_rad)],
        [0, np.sin(el_rad), np.cos(el_rad)]
    ])
    
    # Azimuth - rotation around y-axis
    R_y = np.array([
        [np.cos(az_rad), 0, -np.sin(az_rad)],
        [0, 1, 0],
        [np.sin(az_rad), 0, np.cos(az_rad)]
    ])
    
    # Apply rotations in the specified order
    # Matrix multiplication applies from right to left
    R = R_y @ R_x @ R_z  # This applies: roll, then elevation, then azimuth
    
    # Apply rotation matrix to coordinates
    rotated = R @ coords
    
    # Reshape to original dimensions
    u_rot = rotated[0].reshape(original_shape)
    v_rot = rotated[1].reshape(original_shape)
    w_rot = rotated[2].reshape(original_shape)
    
    # Handle scalar return
    if scalar_input:
        return u_rot.item(), v_rot.item(), w_rot.item()
    
    return u_rot, v_rot, w_rot

def transform_coordinates(pattern_obj, format: str = 'sided') -> None:
    """
    Transform pattern coordinates to conform to a specified theta/phi convention.
    
    This function rearranges the existing pattern data to match one of two standard
    coordinate conventions without interpolation:
    
    - 'sided': theta 0:180, phi 0:360 (spherical convention)
    - 'central': theta -180:180, phi 0:180 (more common for antenna patterns)
    
    Args:
        pattern_obj: AntennaPattern object to modify
        format: Target coordinate format ('sided' or 'central')
            
    Raises:
        ValueError: If format is not 'sided' or 'central'
    """
    
    if format not in ['sided', 'central']:
        raise ValueError("Format must be 'sided' or 'central'")
    
    # Get current coordinates
    theta = pattern_obj.theta_angles
    phi = pattern_obj.phi_angles
    
    # Get field components
    e_theta = pattern_obj.data.e_theta.values
    e_phi = pattern_obj.data.e_phi.values
    frequencies = pattern_obj.frequencies
    
    # Check current format
    theta_min, theta_max = np.min(theta), np.max(theta)
    phi_min, phi_max = np.min(phi), np.max(phi)
    
    is_sided = theta_min >= 0 and theta_max <= 180 and phi_min >= 0 and phi_max <= 360
    is_central = theta_min >= -180 and theta_max <= 180 and phi_min >= 0 and phi_max <= 180
    
    # If already in the correct format, return
    if (format == 'sided' and is_sided) or (format == 'central' and is_central):
        return
    
    # Apply transformation based on target format
    if format == 'sided':
        # Target: theta 0:180, phi 0:360
        # Every cut will now get split into two cuts. 
        # The negative theta values will become a new cut with postitive theta, phi +180 deg.

        # ensure phi is within central range
        if np.max(phi) >= 180:
            raise ValueError("Input phi must be less than 180 when transforming to sided")
        
        # get theta 0 idx
        theta0_idx = np.argmin(np.abs(theta))

        # create new theta vector
        new_theta = theta[theta0_idx:]

        # create new phi vector
        new_phi = np.concatenate((phi, phi+180))

        # initilize new electric field vectors
        new_e_theta = np.zeros(shape=(frequencies.size, new_theta.size, new_phi.size), dtype=np.complex128)
        new_e_phi = np.zeros(shape=(frequencies.size, new_theta.size, new_phi.size), dtype=np.complex128)
        
        # Fill new electric field vectors
        new_e_theta[:, :, :np.size(phi)] = e_theta[:, theta0_idx:, :]
        new_e_theta[:, :, np.size(phi):] = np.flip(e_theta[:, :theta0_idx+1, :], axis=1)
        new_e_phi[:, :, :np.size(phi)] = e_phi[:, theta0_idx:, :]
        new_e_phi[:, :, np.size(phi):] = np.flip(e_phi[:, :theta0_idx+1, :], axis=1)
    
    elif format == 'central':
        # Target: theta -180:180, phi 0:180

        # ensure theta starts at 0
        if theta[0] != 0:
            raise ValueError("Input theta must start at 0 when transforming to central")

        # generate new theta array
        new_theta = np.concatenate((-np.flip(theta[1:]), theta))

        # get phi 180 crossing
        phi180_idx = np.argmax(phi >= 180)

        # generate new phi vector
        new_phi = phi[:phi180_idx]

        # initilize new electric field vectors
        new_e_theta = np.zeros(shape=(frequencies.size, new_theta.size, new_phi.size), dtype=np.complex128)
        new_e_phi = np.zeros(shape=(frequencies.size, new_theta.size, new_phi.size), dtype=np.complex128)

        # Fill new electric field vectors
        new_e_theta[:, :theta.size, :] = np.flip(e_theta[:, :, phi180_idx:], axis=2)
        new_e_theta[:, theta.size-1:, :] = e_theta[:, :, :phi180_idx]
        new_e_phi[:, :theta.size, :] = np.flip(e_phi[:, :, phi180_idx:], axis=2)
        new_e_phi[:, theta.size-1:, :] = e_phi[:, :, :phi180_idx]
    
    # Update pattern data with new coordinates and field components
    pattern_obj.data = pattern_obj.data.assign_coords({
        'theta': new_theta,
        'phi': new_phi
    })
    
    pattern_obj.data['e_theta'] = (['frequency', 'theta', 'phi'], new_e_theta)
    pattern_obj.data['e_phi'] = (['frequency', 'theta', 'phi'], new_e_phi)
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'transform_coordinates',
            'format': format,
            'old_theta_range': [float(theta_min), float(theta_max)],
            'old_phi_range': [float(phi_min), float(phi_max)],
            'new_theta_range': [float(np.min(new_theta)), float(np.max(new_theta))],
            'new_phi_range': [float(np.min(new_phi)), float(np.max(new_phi))]
        })

def mirror_pattern(pattern_obj) -> None:
    """
    Mirror the antenna pattern by flipping the theta axis.
    
    This function flips the pattern data along the theta axis, effectively
    creating a mirrored version of the pattern.
    
    Args:
        pattern_obj: AntennaPattern object to modify
    """
    # Get pattern data
    theta = pattern_obj.data.theta.values
    e_theta = pattern_obj.data.e_theta.values
    e_phi = pattern_obj.data.e_phi.values
    
    # Flip the theta angles
    new_theta = -np.flip(theta)
    
    # Flip e_theta and e_phi along the theta axis
    # For e_theta, we need to negate the values when flipping
    new_e_theta = -np.flip(e_theta, axis=1)
    # For e_phi, we just flip without negation
    new_e_phi = -np.flip(e_phi, axis=1)
    
    # Update pattern data with the mirrored values
    pattern_obj.data['e_theta'].values = new_e_theta
    pattern_obj.data['e_phi'].values = new_e_phi
    
    # Update the theta coordinates
    pattern_obj.data = pattern_obj.data.assign_coords({
        'theta': new_theta
    })
    
    # Recalculate co-pol and cross-pol components
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'mirror_pattern'
        })

def normalize_at_boresight(pattern_obj) -> None:
    """
    Normalize the phase and magnitude of an antenna pattern so that both e_theta 
    and e_phi components for all phi cuts have the same phase and magnitude at 
    boresight (theta=0).
    
    This function:
    1. Finds the boresight (theta=0) point for each phi angle
    2. For e_theta:
       - Calculates average magnitude at boresight across all phi cuts
       - Sets a reference phase (from the first phi cut)
       - Normalizes all phi cuts to have the same magnitude and phase at boresight
    3. For e_phi:
       - Follows the same process independently
    
    This ensures that all cuts cross at a common point at boresight for visualization
    and analysis purposes.
    
    Args:
        pattern_obj: AntennaPattern object to modify
    
    Raises:
        ValueError: If the pattern doesn't have a theta=0 point
    """
    
    # Get underlying numpy arrays
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values.copy()
    e_phi = pattern_obj.data.e_phi.values.copy()
    
    # Find the index for theta=0 (boresight)
    theta0_idx = np.argmin(np.abs(theta))
    
    # Check if we have a point close enough to boresight
    if abs(theta[theta0_idx]) > 1.0:  # Allow for small grid spacing differences
        raise ValueError(f"Pattern doesn't have a point sufficiently close to boresight. "
                         f"Closest point is at theta={theta[theta0_idx]} degrees.")
    
    # Process each frequency separately
    for f_idx in range(len(frequency)):
        # Get e_theta values at boresight for each phi cut
        e_theta_boresight = e_theta[f_idx, theta0_idx, :]
        e_phi_boresight = e_phi[f_idx, theta0_idx, :]
        
        # Calculate magnitude at boresight
        e_theta_magnitude = np.abs(e_theta_boresight)
        e_phi_magnitude = np.abs(e_phi_boresight)
        
        # Calculate the average magnitude at boresight
        e_theta_avg_magnitude = np.mean(e_theta_magnitude)
        e_phi_avg_magnitude = np.mean(e_phi_magnitude)
        
        # Instead of averaging phases,
        # choose the first phi cut's phase as reference for each component
        e_theta_ref_phase = np.angle(e_theta_boresight[0])
        e_phi_ref_phase = np.angle(e_phi_boresight[0])
        
        # Apply normalization to each phi cut
        for p_idx in range(len(phi)):
            # Calculate complex correction factors as direct ratio between
            # reference value and current value at boresight
            e_theta_current = e_theta[f_idx, theta0_idx, p_idx]
            e_phi_current = e_phi[f_idx, theta0_idx, p_idx]

            # Calculate reference complex values at boresight (target values)
            e_theta_ref = e_theta_avg_magnitude * np.exp(1j * (e_theta_ref_phase-np.radians(phi[p_idx])))
            e_phi_ref = e_phi_avg_magnitude * np.exp(1j * (e_phi_ref_phase-np.radians(phi[p_idx])))
                
            # Avoid division by zero
            if abs(e_theta_current) < 1e-15:
                e_theta_current = 1e-15 * (np.cos(e_theta_ref_phase) + 1j * np.sin(e_theta_ref_phase))
            if abs(e_phi_current) < 1e-15:
                e_phi_current = 1e-15 * (np.cos(e_phi_ref_phase) + 1j * np.sin(e_phi_ref_phase))
            
            # Calculate correction as simple ratio of complex numbers
            theta_correction = e_theta_ref / e_theta_current
            phi_correction = e_phi_ref / e_phi_current
            
            # Apply corrections to all theta values for this phi cut
            e_theta[f_idx, :, p_idx] *= theta_correction
            e_phi[f_idx, :, p_idx] *= phi_correction
    
    # Update the pattern data directly
    pattern_obj.data['e_theta'].values = e_theta
    pattern_obj.data['e_phi'].values = e_phi
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'normalize_at_boresight',
            'boresight_theta': float(theta[theta0_idx]),
            'normalized_components': ['e_theta', 'e_phi'],
            'normalized_attributes': ['magnitude', 'phase']
        })

def shift_theta_origin(pattern_obj, theta_offset: float) -> None:
    """
    Shifts the origin of the theta coordinate axis for all phi cuts.
    
    This is useful for aligning measurement data when the mechanical 
    antenna rotation axis doesn't align with the desired coordinate 
    system (e.g., antenna boresight).
    
    The function preserves the original theta grid while shifting 
    the pattern data through interpolation along each phi cut.
    
    Args:
        pattern_obj: AntennaPattern object to modify
        theta_offset: Angle in degrees to shift the theta origin.
                     Positive values move theta=0 to the right (positive theta),
                     negative values move theta=0 to the left (negative theta).
                     
    Notes:
        - This performs interpolation along the theta axis for each phi cut
        - Complex field components are interpolated separately for amplitude and phase
          to avoid interpolation issues with complex numbers
        - Phase discontinuities are handled by unwrapping before interpolation
    """
    # Get underlying numpy arrays
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values.copy()
    e_phi = pattern_obj.data.e_phi.values.copy()
    
    # Create shifted theta array for original data position
    # Positive theta_offset means data shifts left, grid points stay the same
    shifted_theta = theta - theta_offset
    
    # Initialize output arrays with same shape as input
    e_theta_new = np.zeros_like(e_theta, dtype=complex)
    e_phi_new = np.zeros_like(e_phi, dtype=complex)
    
    # Process each frequency and phi cut separately
    for f_idx in range(len(frequency)):
        for p_idx in range(len(phi)):
            # For each component, separate amplitude and phase for interpolation
            
            # Process e_theta
            amp_theta = np.abs(e_theta[f_idx, :, p_idx])
            phase_theta = np.unwrap(np.angle(e_theta[f_idx, :, p_idx]))
            
            # Create interpolation functions for amplitude and phase
            amp_interp_theta = interp1d(
                shifted_theta, 
                amp_theta, 
                kind='cubic', 
                bounds_error=False, 
                fill_value=(amp_theta[0], amp_theta[-1])
            )
            
            phase_interp_theta = interp1d(
                shifted_theta, 
                phase_theta, 
                kind='cubic', 
                bounds_error=False, 
                fill_value=(phase_theta[0], phase_theta[-1])
            )
            
            # Interpolate onto original grid
            amp_new_theta = amp_interp_theta(theta)
            phase_new_theta = phase_interp_theta(theta)
            
            # Combine amplitude and phase back to complex
            e_theta_new[f_idx, :, p_idx] = amp_new_theta * np.exp(1j * phase_new_theta)
            
            # Process e_phi
            amp_phi = np.abs(e_phi[f_idx, :, p_idx])
            phase_phi = np.unwrap(np.angle(e_phi[f_idx, :, p_idx]))
            
            # Create interpolation functions for amplitude and phase
            amp_interp_phi = interp1d(
                shifted_theta, 
                amp_phi, 
                kind='cubic', 
                bounds_error=False, 
                fill_value=(amp_phi[0], amp_phi[-1])
            )
            
            phase_interp_phi = interp1d(
                shifted_theta, 
                phase_phi, 
                kind='cubic', 
                bounds_error=False, 
                fill_value=(phase_phi[0], phase_phi[-1])
            )
            
            # Interpolate onto original grid
            amp_new_phi = amp_interp_phi(theta)
            phase_new_phi = phase_interp_phi(theta)
            
            # Combine amplitude and phase back to complex
            e_phi_new[f_idx, :, p_idx] = amp_new_phi * np.exp(1j * phase_new_phi)
    
    # Update the pattern data
    pattern_obj.data['e_theta'].values = e_theta_new
    pattern_obj.data['e_phi'].values = e_phi_new
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'shift_theta_origin',
            'theta_offset': float(theta_offset)
        })

def shift_phi_origin(pattern_obj, phi_offset: float) -> None:
    """
    Shifts the origin of the phi coordinate axis for the pattern.
    
    This is useful for aligning measurement data when the mechanical 
    antenna rotation reference doesn't align with the desired coordinate
    system (e.g., principal planes).
    
    The function preserves the original phi grid while shifting 
    the pattern data through interpolation. To maintain field properties,
    it converts to Cartesian field components (e_x, e_y) before interpolation,
    then converts back to spherical components (e_theta, e_phi).
    
    Args:
        pattern_obj: AntennaPattern object to modify
        phi_offset: Angle in degrees to shift the phi origin.
                   Positive values rotate phi clockwise,
                   negative values rotate phi counterclockwise.
                   
    Notes:
        - This performs interpolation along the phi axis for each theta value
        - Converts to Cartesian field components for interpolation to preserve field properties
        - Takes into account the periodicity of phi (0° = 360°)
    """
    # Get underlying numpy arrays
    frequency = pattern_obj.data.frequency.values
    theta = pattern_obj.data.theta.values
    phi = pattern_obj.data.phi.values
    e_theta = pattern_obj.data.e_theta.values.copy()
    e_phi = pattern_obj.data.e_phi.values.copy()
    
    # Create shifted phi array for original data position
    # Phi is periodic, so we handle wraparound
    # Note: For interpolation, we need to shift in the opposite direction
    # since we're interpolating from the shifted grid to the original grid
    shifted_phi = np.mod(phi + phi_offset, 360.0)
    
    # Initialize output arrays with same shape as input
    e_theta_new = np.zeros_like(e_theta, dtype=complex)
    e_phi_new = np.zeros_like(e_phi, dtype=complex)
    
    # Process each frequency and theta separately
    for f_idx in range(len(frequency)):
        for t_idx in range(len(theta)):
            # First convert spherical components to Cartesian
            e_x, e_y = polarization_tp2xy(phi, e_theta[f_idx, t_idx, :], e_phi[f_idx, t_idx, :])
            
            # Now separate magnitude and phase for each Cartesian component
            # This approach preserves magnitude better during interpolation
            mag_x = np.abs(e_x)
            phase_x = np.unwrap(np.angle(e_x))
            
            mag_y = np.abs(e_y)
            phase_y = np.unwrap(np.angle(e_y))
            
            # Create extended phi arrays for periodic interpolation
            ext_phi = np.concatenate([phi - 360.0, phi, phi + 360.0])
            ext_shifted_phi = np.concatenate([shifted_phi - 360.0, shifted_phi, shifted_phi + 360.0])
            
            # Extend the magnitude and phase arrays
            ext_mag_x = np.concatenate([mag_x, mag_x, mag_x])
            ext_phase_x = np.concatenate([phase_x - 2*np.pi, phase_x, phase_x + 2*np.pi])
            
            ext_mag_y = np.concatenate([mag_y, mag_y, mag_y])
            ext_phase_y = np.concatenate([phase_y - 2*np.pi, phase_y, phase_y + 2*np.pi])
            
            # Create interpolation functions for magnitude (linear to preserve peaks)
            # and phase (cubic for smoothness)
            mag_x_interp = interp1d(
                ext_shifted_phi, 
                ext_mag_x, 
                kind='linear',  # Linear for magnitude to preserve peaks
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            phase_x_interp = interp1d(
                ext_shifted_phi, 
                ext_phase_x, 
                kind='cubic', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            mag_y_interp = interp1d(
                ext_shifted_phi, 
                ext_mag_y, 
                kind='linear',  # Linear for magnitude to preserve peaks
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            phase_y_interp = interp1d(
                ext_shifted_phi, 
                ext_phase_y, 
                kind='cubic', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            # Interpolate onto original grid
            mag_x_new = mag_x_interp(phi)
            phase_x_new = phase_x_interp(phi)
            
            mag_y_new = mag_y_interp(phi)
            phase_y_new = phase_y_interp(phi)
            
            # Recombine magnitude and phase to get complex values
            e_x_new = mag_x_new * np.exp(1j * phase_x_new)
            e_y_new = mag_y_new * np.exp(1j * phase_y_new)
            
            # Convert back to spherical components
            e_theta_new[f_idx, t_idx, :], e_phi_new[f_idx, t_idx, :] = polarization_xy2pt(
                phi, e_x_new, e_y_new
            )
    
    # Update the pattern data
    pattern_obj.data['e_theta'].values = e_theta_new
    pattern_obj.data['e_phi'].values = e_phi_new
    
    # Recalculate derived components e_co and e_cx
    pattern_obj.assign_polarization(pattern_obj.polarization)
    
    # Clear cache
    pattern_obj.clear_cache()
    
    # Update metadata if needed
    if hasattr(pattern_obj, 'metadata') and pattern_obj.metadata is not None:
        if 'operations' not in pattern_obj.metadata:
            pattern_obj.metadata['operations'] = []
        pattern_obj.metadata['operations'].append({
            'type': 'shift_phi_origin',
            'phi_offset': float(phi_offset)
        })