"""
Polarization conversion functions for antenna radiation patterns.

This module provides mathematical functions for converting between different
polarization representations (spherical polarization basis, Ludwig-3 X and Y, Ludwig-3 circular).
"""
import numpy as np
import logging
from typing import Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for clarity
ComplexArray = Union[np.ndarray]
RealArray = Union[np.ndarray]


def polarization_tp2xy(phi: RealArray, e_theta: ComplexArray, e_phi: ComplexArray) -> Tuple[ComplexArray, ComplexArray]:
    """
    Convert spherical field components to Ludwig's III X and Y field components.
    
    Args:
        phi: Spherical angle phi in degrees
        e_theta: Spherical polarization component theta
        e_phi: Spherical polarization component phi
        
    Returns:
        Tuple[ndarray, ndarray]: Ludwig's III field components e_x and e_y
    """
    # Convert phi to numpy array and to radians
    phi_rad = np.radians(np.asarray(phi))
    
    # Handle broadcasting for different dimensions
    if np.ndim(e_theta) > np.ndim(phi_rad):
        # Add necessary dimensions for broadcasting
        extra_dims = np.ndim(e_theta) - np.ndim(phi_rad)
        phi_rad = np.expand_dims(phi_rad, axis=tuple(range(extra_dims)))
    
    # Vectorized computation
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    
    e_x = cos_phi * e_theta - sin_phi * e_phi
    e_y = sin_phi * e_theta + cos_phi * e_phi
    
    return e_x, e_y


def polarization_xy2pt(phi: RealArray, e_x: ComplexArray, e_y: ComplexArray) -> Tuple[ComplexArray, ComplexArray]:
    """
    Convert Ludwig's III X and Y field components to spherical field components.
    
    Args:
        phi: Spherical angle phi in degrees
        e_x: Ludwig's III co-polarization field component
        e_y: Ludwig's III cross-polarization field component
        
    Returns:
        Tuple[ndarray, ndarray]: Spherical field components e_theta and e_phi
    """
    # Convert phi to radians
    phi_rad = np.radians(np.asarray(phi))
    
    # Handle broadcasting for different dimensions
    if np.ndim(e_x) > np.ndim(phi_rad):
        extra_dims = np.ndim(e_x) - np.ndim(phi_rad)
        phi_rad = np.expand_dims(phi_rad, axis=tuple(range(extra_dims)))
    
    # Vectorized computation
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    
    e_theta = cos_phi * e_x + sin_phi * e_y
    e_phi = -sin_phi * e_x + cos_phi * e_y
    
    return e_theta, e_phi


def polarization_tp2rl(phi: RealArray, e_theta: ComplexArray, e_phi: ComplexArray) -> Tuple[ComplexArray, ComplexArray]:
    """
    Convert spherical polarization basis field components to Ludwig's III circularly polarized components. 
    
    Args:
        phi: Spherical angle phi in degrees
        e_theta: Spherical polarization component theta
        e_phi: Spherical polarization component phi
        
    Returns:
        Tuple[ndarray, ndarray]: Circular field components e_right and e_left
    """
    # First convert to x,y
    e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
    
    # Calculate RHCP and LHCP components
    sqrt2_inv = 1 / np.sqrt(2)
    e_right = sqrt2_inv * (e_x + 1j * e_y)
    e_left = sqrt2_inv * (e_x - 1j * e_y)
    
    return e_right, e_left


def polarization_rl2xy(e_right: ComplexArray, e_left: ComplexArray) -> Tuple[ComplexArray, ComplexArray]:
    """
    Convert Ludwig's III circularly polarized components to Ludwig's III X and Y field components.
    
    Args:
        e_right: Right hand circular field component
        e_left: Left hand circular field component
        
    Returns:
        Tuple[ndarray, ndarray]: Ludwig's III field components e_x and e_y
    """
    sqrt2_inv = 1 / np.sqrt(2)
    e_x = sqrt2_inv * (e_right + e_left)
    e_y = 1j * sqrt2_inv * (e_left - e_right)
    
    return e_x, e_y


def polarization_rl2tp(phi: RealArray, e_right: ComplexArray, e_left: ComplexArray) -> Tuple[ComplexArray, ComplexArray]:
    """
    Convert Ludwig's III circularly polarized components to spherical polarization basis field components.
    
    Args:
        phi: Spherical angle phi in degrees
        e_right: Right hand circular field component
        e_left: Left hand circular field component
        
    Returns:
        Tuple[ndarray, ndarray]: Spherical field components e_theta and e_phi
    """
    # First convert to x,y
    e_x, e_y = polarization_rl2xy(e_right, e_left)
    
    # Then convert to theta,phi
    e_theta, e_phi = polarization_xy2pt(phi, e_x, e_y)
    
    return e_theta, e_phi