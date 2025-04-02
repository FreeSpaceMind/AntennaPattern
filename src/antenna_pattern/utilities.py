"""
Common utility functions and constants for antenna pattern analysis.
"""
import numpy as np
from typing import Tuple, Union, Optional, List, Any

# Physical constants
lightspeed = 299792458  # Speed of light in vacuum (m/s)
freespace_permittivity = 8.8541878128e-12  # Vacuum permittivity (F/m)
freespace_impedance = 376.730313668  # Impedance of free space (Ohms)

# Astronomical constants
moon_radius = 1737.1e3  # Mean radius of the Moon (m)
earth_radius = 6378.14e3  # Mean equatorial radius of the Earth (m)

# Type aliases
NumericArray = Union[np.ndarray, List[float], List[int], Tuple[float, ...], Tuple[int, ...]]


def find_nearest(array: NumericArray, value: float) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
    """
    Find the value in an array that is closest to a specified value and its index.
    
    Args:
        array: Array-like collection of numeric values
        value: Target value to find the nearest element to
    
    Returns:
        Tuple containing (nearest_value, index_of_nearest_value)
        
    Raises:
        ValueError: If input array is empty
    """
    array = np.asarray(array)
    
    if array.size == 0:
        raise ValueError("Input array is empty")
    
    idx = np.abs(array - value).argmin()
    return array[idx], idx


def frequency_to_wavelength(frequency: Union[float, np.ndarray], dielectric_constant: float = 1.0) -> np.ndarray:
    """
    Convert frequency to wavelength.
    
    Args:
        frequency: Frequency in Hz
        dielectric_constant: Relative permittivity of the medium (default: 1.0 for vacuum)
    
    Returns:
        Wavelength in meters
        
    Raises:
        ValueError: If frequency is zero or negative
        ValueError: If dielectric constant is negative
    """
    # Convert input to numpy array if it's not already
    if not isinstance(frequency, np.ndarray):
        frequency = np.asarray(frequency)
    
    # Validate inputs
    if np.any(frequency <= 0):
        raise ValueError("Frequency must be positive")
    
    if dielectric_constant < 0:
        raise ValueError("Dielectric constant must be non-negative")
    
    return lightspeed / (frequency * np.sqrt(dielectric_constant))


def wavelength_to_frequency(wavelength: Union[float, np.ndarray], dielectric_constant: float = 1.0) -> np.ndarray:
    """
    Convert wavelength to frequency.
    
    Args:
        wavelength: Wavelength in meters
        dielectric_constant: Relative permittivity of the medium (default: 1.0 for vacuum)
    
    Returns:
        Frequency in Hz
        
    Raises:
        ValueError: If wavelength is zero or negative
        ValueError: If dielectric constant is negative
    """
    # Convert input to numpy array if it's not already
    if not isinstance(wavelength, np.ndarray):
        wavelength = np.asarray(wavelength)
    
    # Validate inputs
    if np.any(wavelength <= 0):
        raise ValueError("Wavelength must be positive")
    
    if dielectric_constant < 0:
        raise ValueError("Dielectric constant must be non-negative")
    
    return lightspeed / (wavelength * np.sqrt(dielectric_constant))


def db_to_linear(db_value: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert a dB value to linear scale.
    
    Args:
        db_value: Value in dB
    
    Returns:
        Value in linear scale
    """
    if not isinstance(db_value, np.ndarray):
        db_value = np.asarray(db_value)
        
    return 10 ** (db_value / 10.0)


def linear_to_db(linear_value: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert a linear value to dB scale.
    
    Args:
        linear_value: Value in linear scale
    
    Returns:
        Value in dB
        
    Raises:
        ValueError: If linear value is negative
    """
    if not isinstance(linear_value, np.ndarray):
        linear_value = np.asarray(linear_value)
    
    if np.any(linear_value < 0):
        raise ValueError("Linear values must be non-negative for dB conversion")
    
    # Set values close to zero to a small positive number to avoid log(0)
    linear_value = np.maximum(linear_value, 1e-15)
    
    return 10.0 * np.log10(linear_value)


def unwrap_phase(phase: np.ndarray, discont: float = np.pi) -> np.ndarray:
    """
    Enhanced phase unwrapping with adjustable discontinuity threshold.
    
    Args:
        phase: Array of phase values in radians
        discont: Size of the discontinuity for unwrapping (default: π)
    
    Returns:
        Unwrapped phase array
    """
    return np.unwrap(phase, discont=discont)


def interpolate_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """
    Linearly interpolate to find the x value where y crosses a threshold.
    
    Args:
        x: Array of x coordinates (size 2)
        y: Array of y coordinates (size 2)
        threshold: The y value to find the crossing for
    
    Returns:
        Interpolated x value at the crossing
    """
    return x[0] + (threshold - y[0]) * (x[1] - x[0]) / (y[1] - y[0])


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