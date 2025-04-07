"""
Functions to accompany the AntennaPattern class. Moved here to keep the main class short.
"""
import numpy as np
from typing import Tuple, Union, Optional, List, Any, Callable

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