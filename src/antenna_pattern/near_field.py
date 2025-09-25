"""
Optimized planar near-field to far-field transformation using plane wave spectrum method.
"""

import numpy as np
from scipy.signal import windows
from typing import Tuple, Optional, Literal
from antenna_pattern import AntennaPattern


def nf_to_ff_planar(
    E_x: np.ndarray,
    E_y: np.ndarray, 
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    frequencies: np.ndarray,
    scan_distance: float,
    coordinate_system: Literal['central', 'sided'] = 'central',
    window_type: str = 'tukey',
    window_param: float = 0.25,
    zero_pad_factor: int = 2,
    theta_max: float = 90.0,
    theta_spacing: Optional[float] = None,
    phi_spacing: Optional[float] = None
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
        coordinate_system: 'central' (-theta_max to +theta_max) or 'sided' (0 to theta_max)
        window_type: Windowing function ('tukey', 'hann', 'hamming', or 'none')
        window_param: Window parameter (for tukey: cosine fraction)
        zero_pad_factor: Factor for zero padding (1 = no padding, 2 = double size)
        theta_max: Maximum theta angle for far-field pattern in degrees
        theta_spacing: Output theta step size in degrees (auto if None)
        phi_spacing: Output phi step size in degrees (auto if None)
        
    Returns:
        AntennaPattern: Far-field pattern object
    """
    
    print(f"Starting NF-FF transformation...")
    print(f"Input shape: {E_x.shape}")
    print(f"Coordinate system: {coordinate_system}")
    print(f"Window: {window_type}, Zero pad factor: {zero_pad_factor}")
    
    # Validate inputs
    if E_x.shape != E_y.shape:
        raise ValueError("E_x and E_y must have the same shape")
    
    M, N, F = E_x.shape
    
    if len(x_positions) != M or len(y_positions) != N or len(frequencies) != F:
        raise ValueError("Position and frequency arrays must match field array dimensions")
    
    # Calculate sampling parameters
    dx = x_positions[1] - x_positions[0] if M > 1 else 0.1
    dy = y_positions[1] - y_positions[0] if N > 1 else 0.1
    
    print(f"Sample spacing: dx = {dx:.4f} m, dy = {dy:.4f} m")
    
    # Check for uniform sampling
    if M > 1 and not np.allclose(np.diff(x_positions), dx, rtol=1e-6):
        raise ValueError("X positions must be uniformly sampled")
    if N > 1 and not np.allclose(np.diff(y_positions), dy, rtol=1e-6):
        raise ValueError("Y positions must be uniformly sampled")
    
    # Create output angle grids
    if coordinate_system == 'central':
        theta_min, theta_max_range = -theta_max, theta_max
        phi_min, phi_max_range = 0, 180  # Only need 0 to 180 for central
    elif coordinate_system == 'sided':
        theta_min, theta_max_range = 0, theta_max
        phi_min, phi_max_range = 0, 360  # Full 360 for sided
    else:
        raise ValueError(f"Unknown coordinate_system: {coordinate_system}")
    
    # Set default spacing
    if theta_spacing is None:
        theta_spacing = 1.0
    if phi_spacing is None:
        phi_spacing = 2.0
        
    output_theta = np.arange(theta_min, theta_max_range + theta_spacing, theta_spacing)
    output_phi = np.arange(phi_min, phi_max_range, phi_spacing)
    
    print(f"Output grid: {len(output_theta)} × {len(output_phi)} points")
    print(f"Theta: {output_theta[0]:.1f}° to {output_theta[-1]:.1f}°")
    print(f"Phi: {output_phi[0]:.1f}° to {output_phi[-1]:.1f}°")
    
    # Apply windowing
    print("Applying windowing...")
    window_x, window_y = _create_2d_window(M, N, window_type, window_param)
    
    # Calculate zero-padded dimensions
    M_pad = M * zero_pad_factor
    N_pad = N * zero_pad_factor
    
    print(f"Zero-padded FFT size: {M_pad} × {N_pad}")
    
    # Initialize output arrays
    E_theta_ff = np.zeros((F, len(output_theta), len(output_phi)), dtype=complex)
    E_phi_ff = np.zeros((F, len(output_theta), len(output_phi)), dtype=complex)
    
    # Pre-calculate k-space grids for all frequencies
    print("Pre-calculating k-space grids...")
    # Increase k-space resolution to support fine angular sampling
    kx_max = min(np.pi / dx, 2.0 * 2 * np.pi * frequencies.max() / 3e8)  # Reduced from 3.0
    ky_max = min(np.pi / dy, 2.0 * 2 * np.pi * frequencies.max() / 3e8)
    kx = np.linspace(-kx_max, kx_max, M_pad)
    ky = np.linspace(-ky_max, ky_max, N_pad)
    
    # Calculate actual k-space resolution
    dk_x = kx[1] - kx[0] if len(kx) > 1 else 0
    dk_y = ky[1] - ky[0] if len(ky) > 1 else 0
    print(f"K-space resolution: dkx = {dk_x:.4f}, dky = {dk_y:.4f} rad/m")
    
    # Process each frequency
    for freq_idx, freq in enumerate(frequencies):
        print(f"Processing frequency {freq_idx+1}/{F}: {freq/1e9:.3f} GHz")
        
        k0 = 2 * np.pi * freq / 3e8
        wavelength = 3e8 / freq
        
        # Check sampling
        if dx > wavelength/2:
            print(f"  WARNING: X undersampled! dx={dx:.4f} m > λ/2={wavelength/2:.4f} m")
        if dy > wavelength/2:
            print(f"  WARNING: Y undersampled! dy={dy:.4f} m > λ/2={wavelength/2:.4f} m")
        
        # Get fields at this frequency and apply window + phase correction
        Ex_windowed = E_x[:, :, freq_idx] * window_x * window_y
        Ey_windowed = E_y[:, :, freq_idx] * window_x * window_y
        
        phase_correction = np.exp(1j * k0 * scan_distance)
        Ex_windowed *= phase_correction
        Ey_windowed *= phase_correction
        
        # Zero pad and perform 2D FFT
        Ex_padded = np.zeros((M_pad, N_pad), dtype=complex)
        Ey_padded = np.zeros((M_pad, N_pad), dtype=complex)
        
        start_m = (M_pad - M) // 2
        start_n = (N_pad - N) // 2
        Ex_padded[start_m:start_m+M, start_n:start_n+N] = Ex_windowed
        Ey_padded[start_m:start_m+M, start_n:start_n+N] = Ey_windowed
        
        # 2D FFT to get plane wave spectrum
        Ex_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex_padded)))
        Ey_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ey_padded)))
        
        # Apply proper scaling
        scale_factor = dx * dy / (4 * np.pi**2) * k0
        Ex_spectrum *= scale_factor
        Ey_spectrum *= scale_factor
        
        # Map to output angles with bilinear interpolation
        print(f"  Mapping to output angles...")
        for i, theta in enumerate(output_theta):
            for j, phi in enumerate(output_phi):
                # Convert angle to k-space
                kx_target, ky_target = _angles_to_kspace(theta, phi, k0, coordinate_system)
                
                # Skip evanescent waves
                if kx_target**2 + ky_target**2 > k0**2:
                    continue
                
                # Bilinear interpolation in k-space
                Ex_val = _bilinear_interp(Ex_spectrum, kx, ky, kx_target, ky_target)
                Ey_val = _bilinear_interp(Ey_spectrum, kx, ky, kx_target, ky_target)
                
                # Transform to spherical components
                E_theta_val, E_phi_val = _xy_to_spherical_ff(Ex_val, Ey_val, theta, phi)
                
                E_theta_ff[freq_idx, i, j] = E_theta_val
                E_phi_ff[freq_idx, i, j] = E_phi_val
    
    print("Creating AntennaPattern object...")
    
    # Create AntennaPattern object
    return AntennaPattern(
        theta=output_theta,
        phi=output_phi, 
        frequency=frequencies,
        e_theta=E_theta_ff,
        e_phi=E_phi_ff
    )


def _create_2d_window(M: int, N: int, window_type: str, window_param: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create 2D separable window function."""
    if window_type.lower() == 'none':
        return np.ones((M, 1)), np.ones((1, N))
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


def _bilinear_interp(data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, 
                    x_target: float, y_target: float) -> complex:
    """Fast bilinear interpolation in 2D grid with boundary handling."""
    # Find bracketing indices
    x_idx = np.searchsorted(x_coords, x_target)
    y_idx = np.searchsorted(y_coords, y_target)
    
    # Handle boundary conditions more carefully
    if x_idx == 0:
        x_idx = 1
    elif x_idx >= len(x_coords):
        x_idx = len(x_coords) - 1
        
    if y_idx == 0:
        y_idx = 1
    elif y_idx >= len(y_coords):
        y_idx = len(y_coords) - 1
    
    # Check if we're exactly on a grid point
    if (abs(x_target - x_coords[x_idx-1]) < 1e-12 and 
        abs(y_target - y_coords[y_idx-1]) < 1e-12):
        return data[x_idx-1, y_idx-1]
    
    # Get bracketing coordinates and values
    x0, x1 = x_coords[x_idx-1], x_coords[x_idx]
    y0, y1 = y_coords[y_idx-1], y_coords[y_idx]
    
    # Avoid division by zero
    if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
        return data[x_idx-1, y_idx-1]
    
    # Get data values at corners
    f00 = data[x_idx-1, y_idx-1]
    f10 = data[x_idx, y_idx-1] 
    f01 = data[x_idx-1, y_idx]
    f11 = data[x_idx, y_idx]
    
    # Bilinear interpolation weights
    wx = (x_target - x0) / (x1 - x0)
    wy = (y_target - y0) / (y1 - y0)
    
    # Clamp weights to [0,1] for safety
    wx = max(0, min(1, wx))
    wy = max(0, min(1, wy))
    
    # Interpolate
    f_interp = (f00 * (1 - wx) * (1 - wy) + 
                f10 * wx * (1 - wy) + 
                f01 * (1 - wx) * wy + 
                f11 * wx * wy)
    
    return f_interp


def _angles_to_kspace(theta: float, phi: float, k0: float, coord_system: str) -> Tuple[float, float]:
    """Convert angles to k-space coordinates."""
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    if coord_system == 'central':
        # Central coordinates: theta is elevation angle
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)
        
        kx = k0 * sin_theta * cos_phi
        ky = k0 * sin_theta * sin_phi
        
    elif coord_system == 'sided':
        # Sided coordinates: standard spherical
        kx = k0 * np.sin(theta_rad) * np.cos(phi_rad)
        ky = k0 * np.sin(theta_rad) * np.sin(phi_rad)
        
    else:
        raise ValueError(f"Unknown coordinate system: {coord_system}")
    
    return kx, ky


def _xy_to_spherical_ff(Ex: complex, Ey: complex, theta: float, phi: float) -> Tuple[complex, complex]:
    """Transform x,y field components to theta,phi components in far field."""
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad) 
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    
    # Standard far-field transformation
    E_theta = cos_theta * cos_phi * Ex + cos_theta * sin_phi * Ey
    E_phi = -sin_phi * Ex + cos_phi * Ey
    
    return E_theta, E_phi


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