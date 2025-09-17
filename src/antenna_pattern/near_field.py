"""
Near-field to far-field transformation using plane wave spectrum method.

This module provides functionality to transform planar near-field antenna measurements
to far-field patterns using the standard plane wave spectrum approach.
"""

import numpy as np
from scipy.signal import windows
from typing import Tuple, Optional, Union
from antenna_pattern import AntennaPattern


def nf_to_ff_planar(
    E_x: np.ndarray,
    E_y: np.ndarray, 
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    frequencies: np.ndarray,
    scan_distance: float,
    window_type: str = 'tukey',
    window_param: float = 0.25,
    zero_pad_factor: int = 2,
    max_theta: float = 90.0
) -> AntennaPattern:
    """
    Transform planar near-field data to far-field pattern using plane wave spectrum method.
    
    Args:
        E_x: Complex E-field x-component, shape (M, N, F) where M=x_points, N=y_points, F=frequencies
        E_y: Complex E-field y-component, shape (M, N, F)
        x_positions: X-position array in meters, shape (M,)
        y_positions: Y-position array in meters, shape (N,)
        frequencies: Frequency array in Hz, shape (F,)
        scan_distance: Distance from antenna to scan plane in meters
        window_type: Windowing function ('tukey', 'hann', 'hamming', or 'none')
        window_param: Window parameter (for tukey: cosine fraction, others: ignored)
        zero_pad_factor: Factor for zero padding (1 = no padding, 2 = double size)
        max_theta: Maximum theta angle for far-field pattern in degrees
        
    Returns:
        AntennaPattern: Far-field pattern object
    """
    
    # Validate inputs
    if E_x.shape != E_y.shape:
        raise ValueError("E_x and E_y must have the same shape")
    
    M, N, F = E_x.shape
    
    if len(x_positions) != M or len(y_positions) != N or len(frequencies) != F:
        raise ValueError("Position and frequency arrays must match field array dimensions")
    
    # Calculate sampling parameters
    dx = x_positions[1] - x_positions[0] if M > 1 else 0.1  # Default spacing if single point
    dy = y_positions[1] - y_positions[0] if N > 1 else 0.1
    
    # Check for uniform sampling
    if M > 1 and not np.allclose(np.diff(x_positions), dx, rtol=1e-6):
        raise ValueError("X positions must be uniformly sampled")
    if N > 1 and not np.allclose(np.diff(y_positions), dy, rtol=1e-6):
        raise ValueError("Y positions must be uniformly sampled")
    
    # Apply windowing
    window_x, window_y = _create_2d_window(M, N, window_type, window_param)
    
    # Calculate zero-padded dimensions
    M_pad = M * zero_pad_factor
    N_pad = N * zero_pad_factor
    
    # Initialize output arrays
    theta_angles = []
    phi_angles = []
    E_theta_ff = []
    E_phi_ff = []
    
    # Process each frequency
    for freq_idx, freq in enumerate(frequencies):
        # Get fields at this frequency and apply window
        Ex_windowed = E_x[:, :, freq_idx] * window_x * window_y
        Ey_windowed = E_y[:, :, freq_idx] * window_x * window_y
        
        # Apply phase correction for scan plane distance
        k0 = 2 * np.pi * freq / 3e8  # Free space wavenumber
        phase_correction = np.exp(1j * k0 * scan_distance)
        Ex_windowed *= phase_correction
        Ey_windowed *= phase_correction
        
        # Zero pad and perform 2D FFT
        Ex_padded = np.zeros((M_pad, N_pad), dtype=complex)
        Ey_padded = np.zeros((M_pad, N_pad), dtype=complex)
        
        # Center the data in the padded array
        start_m = (M_pad - M) // 2
        start_n = (N_pad - N) // 2
        Ex_padded[start_m:start_m+M, start_n:start_n+N] = Ex_windowed
        Ey_padded[start_m:start_m+M, start_n:start_n+N] = Ey_windowed
        
        # 2D FFT to get plane wave spectrum
        Ex_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex_padded)))
        Ey_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ey_padded)))
        
        # Calculate k-space coordinates
        kx_max = np.pi / dx
        ky_max = np.pi / dy
        kx = np.linspace(-kx_max, kx_max, M_pad)
        ky = np.linspace(-ky_max, ky_max, N_pad)
        
        # Apply proper scaling
        scale_factor = dx * dy / (4 * np.pi**2) * k0
        Ex_spectrum *= scale_factor
        Ey_spectrum *= scale_factor
        
        # Convert to angular coordinates and far-field components
        if freq_idx == 0:  # Only calculate angles once
            theta_grid, phi_grid, valid_mask = _kspace_to_angles(kx, ky, k0, max_theta)
            theta_angles = theta_grid[valid_mask]
            phi_angles = phi_grid[valid_mask]
        
        # Transform to spherical components
        E_theta_temp, E_phi_temp = _xy_to_spherical_ff(
            Ex_spectrum, Ey_spectrum, theta_grid, phi_grid, valid_mask
        )
        
        E_theta_ff.append(E_theta_temp)
        E_phi_ff.append(E_phi_temp)
    
    # Convert to arrays and reshape for AntennaPattern
    E_theta_array = np.array(E_theta_ff)  # Shape: (F, valid_points)
    E_phi_array = np.array(E_phi_ff)     # Shape: (F, valid_points)
    
    # Create unique sorted angle arrays
    unique_theta = np.unique(np.round(theta_angles, 2))
    unique_phi = np.unique(np.round(phi_angles, 2))
    
    # Interpolate onto regular grid
    E_theta_grid, E_phi_grid = _interpolate_to_regular_grid(
        theta_angles, phi_angles, E_theta_array, E_phi_array,
        unique_theta, unique_phi
    )
    
    # Create AntennaPattern object
    return AntennaPattern(
        theta=unique_theta,
        phi=unique_phi, 
        frequency=frequencies,
        e_theta=E_theta_grid,
        e_phi=E_phi_grid
    )


def _create_2d_window(M: int, N: int, window_type: str, window_param: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create 2D separable window function."""
    if window_type.lower() == 'none':
        return np.ones(M), np.ones(N)
    elif window_type.lower() == 'tukey':
        win_x = windows.tukey(M, window_param) if M > 1 else np.ones(1)
        win_y = windows.tukey(N, window_param) if N > 1 else np.ones(1)
    elif window_type.lower() == 'hann':
        win_x = windows.hann(M) if M > 1 else np.ones(1)
        win_y = windows.hann(N) if N > 1 else np.ones(1)
    elif window_type.lower() == 'hamming':
        win_x = windows.hamming(M) if M > 1 else np.ones(1)
        win_y = windows.hamming(N) if N > 1 else np.ones(1)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    return win_x[:, np.newaxis], win_y[np.newaxis, :]


def _kspace_to_angles(kx: np.ndarray, ky: np.ndarray, k0: float, max_theta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert k-space coordinates to angular coordinates."""
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Calculate kz for propagating waves only
    k_rho_sq = KX**2 + KY**2
    valid_prop = k_rho_sq <= k0**2  # Only propagating waves
    
    # Calculate angles
    sin_theta = np.sqrt(k_rho_sq) / k0
    sin_theta = np.clip(sin_theta, 0, 1)  # Ensure physical values
    
    theta = np.rad2deg(np.arcsin(sin_theta))
    phi = np.rad2deg(np.arctan2(KY, KX))
    
    # Apply maximum theta constraint
    theta_valid = theta <= max_theta
    
    # Combined validity mask
    valid_mask = valid_prop & theta_valid
    
    return theta, phi, valid_mask


def _xy_to_spherical_ff(Ex: np.ndarray, Ey: np.ndarray, theta: np.ndarray, 
                       phi: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform x,y field components to theta,phi components in far field."""
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    
    # Transformation matrix elements for far field
    # E_theta = cos(theta)*cos(phi)*Ex + cos(theta)*sin(phi)*Ey - sin(theta)*Ez
    # E_phi = -sin(phi)*Ex + cos(phi)*Ey
    # Note: Ez = 0 for fields in xy plane
    
    E_theta = cos_theta * cos_phi * Ex + cos_theta * sin_phi * Ey
    E_phi = -sin_phi * Ex + cos_phi * Ey
    
    return E_theta[valid_mask], E_phi[valid_mask]


def _interpolate_to_regular_grid(theta_pts: np.ndarray, phi_pts: np.ndarray,
                               E_theta_data: np.ndarray, E_phi_data: np.ndarray,
                               theta_grid: np.ndarray, phi_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate scattered data onto regular theta-phi grid."""
    from scipy.interpolate import griddata
    
    # Create meshgrid for regular output
    THETA_GRID, PHI_GRID = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    # Prepare output arrays
    F = E_theta_data.shape[0]
    E_theta_interp = np.zeros((F, len(theta_grid), len(phi_grid)), dtype=complex)
    E_phi_interp = np.zeros((F, len(theta_grid), len(phi_grid)), dtype=complex)
    
    # Interpolate each frequency
    for freq_idx in range(F):
        # Combine theta, phi points for interpolation
        points = np.column_stack([theta_pts, phi_pts])
        grid_points = np.column_stack([THETA_GRID.ravel(), PHI_GRID.ravel()])
        
        # Interpolate real and imaginary parts separately
        E_theta_real = griddata(points, E_theta_data[freq_idx].real, grid_points, method='linear', fill_value=0)
        E_theta_imag = griddata(points, E_theta_data[freq_idx].imag, grid_points, method='linear', fill_value=0)
        
        E_phi_real = griddata(points, E_phi_data[freq_idx].real, grid_points, method='linear', fill_value=0)
        E_phi_imag = griddata(points, E_phi_data[freq_idx].imag, grid_points, method='linear', fill_value=0)
        
        # Reshape and combine
        E_theta_interp[freq_idx] = (E_theta_real + 1j * E_theta_imag).reshape(len(theta_grid), len(phi_grid))
        E_phi_interp[freq_idx] = (E_phi_real + 1j * E_phi_imag).reshape(len(theta_grid), len(phi_grid))
    
    return E_theta_interp, E_phi_interp


def estimate_scan_requirements(frequency: float, antenna_size: float, desired_theta_max: float = 60.0) -> dict:
    """
    Estimate scan area and sampling requirements for near-field measurement.
    
    Args:
        frequency: Frequency in Hz
        antenna_size: Maximum antenna dimension in meters
        desired_theta_max: Desired maximum theta angle for far-field pattern
        
    Returns:
        Dictionary with scan requirements
    """
    wavelength = 3e8 / frequency
    
    # Rule of thumb: scan area should extend 2-3 wavelengths beyond antenna edges
    scan_margin = 3 * wavelength
    min_scan_size = antenna_size + 2 * scan_margin
    
    # Sampling requirement: λ/2 or better
    max_sample_spacing = wavelength / 2
    min_samples = int(np.ceil(min_scan_size / max_sample_spacing))
    
    # Angular resolution in far field
    angular_resolution = np.rad2deg(wavelength / min_scan_size)
    
    return {
        'min_scan_size_m': min_scan_size,
        'max_sample_spacing_m': max_sample_spacing, 
        'min_samples_per_axis': min_samples,
        'angular_resolution_deg': angular_resolution,
        'wavelength_m': wavelength
    }