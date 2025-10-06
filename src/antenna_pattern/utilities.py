"""
Common utility functions and constants for antenna pattern analysis.
"""
import numpy as np
from typing import Tuple, Union, Optional, List, Any, Callable

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

def create_synthetic_pattern(
    frequencies: np.ndarray,
    theta_angles: np.ndarray,
    phi_angles: np.ndarray,
    peak_gain_dbi: Union[float, np.ndarray],
    polarization: str = 'rhcp',
    beamwidth_deg: Union[float, np.ndarray] = 30.0,
    axial_ratio_db: Union[float, np.ndarray, Callable] = 0.0,
    front_to_back_db: Union[float, np.ndarray] = 25.0,
    sidelobe_level_db: Union[float, np.ndarray] = -20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic antenna pattern based on key antenna parameters.
    
    This function generates an example e_theta and e_phi field components based on
    common antenna parameters, allowing easy creation of test patterns without
    requiring detailed electromagnetic modeling.

    This is primarily intended to be used for rapid testing of code, not to 
    represent a real antenna.
    
    Args:
        frequencies: Array of frequencies in Hz
        theta_angles: Array of theta angles in degrees (-90 to 90 for elevation cuts)
        phi_angles: Array of phi angles in degrees (0-360 for azimuth cuts)
        peak_gain_dbi: Peak gain in dBi. Can be single value or array for each frequency
        polarization: Desired polarization ('rhcp', 'lhcp', 'x', 'y', 'theta', 'phi')
        beamwidth_deg: 3dB beamwidth in degrees. Can be single value or array for each frequency
        axial_ratio_db: Axial ratio in dB. Can be:
            - Single value for constant axial ratio
            - Array matching frequencies for frequency-dependent axial ratio
            - Function taking theta angle (deg) that returns axial ratio (dB)
        front_to_back_db: Front-to-back ratio in dB. Can be single value or array for each frequency
        sidelobe_level_db: Relative sidelobe level in dB. Can be single value or array for each frequency
        
    Returns:
        Tuple of (e_theta, e_phi) complex arrays with shape [freq, theta, phi]
        
    Note:
        To use this pattern, create an AntennaPattern with these components:
        ```
        e_theta, e_phi = create_synthetic_pattern(...)
        pattern = AntennaPattern(
            theta=theta_angles,
            phi=phi_angles,
            frequency=frequencies,
            e_theta=e_theta,
            e_phi=e_phi,
            polarization=polarization  # Optional - will be auto-detected if omitted
        )
        ```
    """
    # Convert inputs to arrays if needed
    if np.isscalar(peak_gain_dbi):
        peak_gain_dbi = np.full(len(frequencies), peak_gain_dbi)
    elif len(peak_gain_dbi) != len(frequencies):
        raise ValueError(f"peak_gain_dbi length ({len(peak_gain_dbi)}) must match frequencies length ({len(frequencies)})")
        
    if np.isscalar(beamwidth_deg):
        beamwidth_deg = np.full(len(frequencies), beamwidth_deg)
    elif len(beamwidth_deg) != len(frequencies):
        raise ValueError(f"beamwidth_deg length ({len(beamwidth_deg)}) must match frequencies length ({len(frequencies)})")
        
    if np.isscalar(front_to_back_db):
        front_to_back_db = np.full(len(frequencies), front_to_back_db)
    elif len(front_to_back_db) != len(frequencies):
        raise ValueError(f"front_to_back_db length ({len(front_to_back_db)}) must match frequencies length ({len(frequencies)})")
        
    if np.isscalar(sidelobe_level_db):
        sidelobe_level_db = np.full(len(frequencies), sidelobe_level_db)
    elif len(sidelobe_level_db) != len(frequencies):
        raise ValueError(f"sidelobe_level_db length ({len(sidelobe_level_db)}) must match frequencies length ({len(frequencies)})")
    
    # Initialize arrays for field components
    e_theta = np.zeros((len(frequencies), len(theta_angles), len(phi_angles)), dtype=complex)
    e_phi = np.zeros((len(frequencies), len(theta_angles), len(phi_angles)), dtype=complex)
    
    # Pattern generation for each frequency
    for freq_idx, freq in enumerate(frequencies):
        # Convert peak gain to linear scale
        peak_gain_linear = 10**(peak_gain_dbi[freq_idx]/10)
        
        # Convert beamwidth to pattern parameter (sigma for Gaussian, parameter for cos^n)
        sigma_deg = beamwidth_deg[freq_idx] / (2 * np.sqrt(2 * np.log(2)))  # Convert HPBW to Gaussian sigma
        cos_n = -np.log(0.5) / np.log(np.cos(np.radians(beamwidth_deg[freq_idx]/2)))  # Power for cosine pattern
        
        # Front-to-back ratio as linear scale
        ftb_linear = 10**(front_to_back_db[freq_idx]/10)
        back_level = 1 / ftb_linear
        
        # Sidelobe level as linear scale relative to peak
        sl_linear = 10**(sidelobe_level_db[freq_idx]/20)  # Convert from dB to amplitude ratio
        
        # Generate pattern for each phi cut
        for phi_idx, phi_val in enumerate(phi_angles):
            # Calculate phi-dependent factors (for patterns with different E/H plane beamwidths)
            # This gives slightly wider beamwidth in H-plane compared to E-plane
            phi_factor = 1.0 + 0.1 * np.cos(np.radians(2 * phi_val))
            pattern_sigma = sigma_deg * phi_factor
            
            for theta_idx, theta_val in enumerate(theta_angles):
                # Calculate normalized angle from boresight
                norm_angle = abs(theta_val) / beamwidth_deg[freq_idx]
                
                # Generate main beam pattern shape
                # Use a modified Gaussian pattern with smoother falloff
                main_beam = np.exp(-(theta_val**2) / (2 * pattern_sigma**2))
                
                # Apply a slight softening to the beam edge to eliminate any discontinuities
                edge_softening = 0.02 * np.exp(-(theta_val**2 - (beamwidth_deg[freq_idx]/1.5)**2)**2 / (beamwidth_deg[freq_idx]**2))
                main_beam = main_beam + edge_softening
                
                # Create a smooth blended pattern of main beam and sidelobes
                
                # First null position based on beamwidth
                null_pos = beamwidth_deg[freq_idx] * 1.5
                
                # Create sidelobe pattern using sinc function
                x = theta_val / null_pos
                if x != 0:
                    sidelobe = sl_linear * abs(np.sin(np.pi * x) / (np.pi * x))
                    
                    # Apply damping to reduce far sidelobes
                    sidelobe *= np.exp(-(theta_val**2) / (2 * (4*pattern_sigma)**2))
                else:
                    sidelobe = 0
                
                # Use analytical function to create continuous pattern
                # This ensures smoothness of both the function and its derivatives
                
                # Determine a continuous transition based on the beamwidth 
                # The width parameter controls the transition zone width
                width = beamwidth_deg[freq_idx] * 0.3  
                
                # Create a smooth continuous weighting using tanh function
                # This ensures C∞ continuity (all derivatives are continuous)
                w_mainbeam = 0.5 * (1 - np.tanh((abs(theta_val) - beamwidth_deg[freq_idx]*0.7) / width))
                w_sidelobe = 1 - w_mainbeam
                
                # Create final pattern with perfectly smooth transition between regions
                pattern_shape = main_beam * w_mainbeam + sidelobe * w_sidelobe
                
                # Ensure pattern never has abrupt fluctuations
                pattern_shape = pattern_shape * (0.998 + 0.002 * np.cos(np.pi * theta_val / 180))
                
                # Apply front-to-back effect - reduce gain in back hemisphere
                if abs(theta_val) > 90:
                    back_angle = 180 - abs(theta_val)
                    # Smooth transition to back level
                    reduction = back_level + (1 - back_level) * np.exp(-(back_angle**2) / 100)
                    pattern_shape *= reduction
                
                # Apply peak gain to get absolute field magnitude
                amplitude = np.sqrt(peak_gain_linear) * pattern_shape
                
                # Calculate axial ratio - support different modes of specification
                if callable(axial_ratio_db):
                    # Function of theta angle
                    ar_db = axial_ratio_db(theta_val)
                elif np.isscalar(axial_ratio_db):
                    # Constant value
                    ar_db = axial_ratio_db
                elif len(axial_ratio_db) == len(frequencies):
                    # Value per frequency
                    ar_db = axial_ratio_db[freq_idx]
                else:
                    # Default good circular polarization
                    ar_db = 0.0
                
                # Convert axial ratio to linear scale for field component calculation
                ar_linear = 10**(ar_db/20)
                
                # Set phase based on phi angle
                phi_phase = np.radians(phi_val)
                
                # Apply polarization and axial ratio to field components
                if polarization.lower() in ['rhcp', 'rh', 'r']:
                    # For RHCP with specified axial ratio
                    if ar_linear > 1.0:
                        # Elliptical polarization - major axis
                        e_theta[freq_idx, theta_idx, phi_idx] = amplitude * ar_linear * np.exp(1j * phi_phase)
                        # Elliptical polarization - minor axis with 90° phase
                        e_phi[freq_idx, theta_idx, phi_idx] = amplitude * np.exp(1j * (phi_phase + np.pi/2))
                    else:
                        # Maintain total power with different AR definition
                        e_theta[freq_idx, theta_idx, phi_idx] = amplitude * np.exp(1j * phi_phase)
                        e_phi[freq_idx, theta_idx, phi_idx] = amplitude * (1/ar_linear) * np.exp(1j * (phi_phase + np.pi/2))
                
                elif polarization.lower() in ['lhcp', 'lh', 'l']:
                    # For LHCP with specified axial ratio
                    if ar_linear > 1.0:
                        # Elliptical polarization - major axis
                        e_theta[freq_idx, theta_idx, phi_idx] = amplitude * ar_linear * np.exp(1j * phi_phase)
                        # Elliptical polarization - minor axis with -90° phase
                        e_phi[freq_idx, theta_idx, phi_idx] = amplitude * np.exp(1j * (phi_phase - np.pi/2))
                    else:
                        # Maintain total power with different AR definition
                        e_theta[freq_idx, theta_idx, phi_idx] = amplitude * np.exp(1j * phi_phase)
                        e_phi[freq_idx, theta_idx, phi_idx] = amplitude * (1/ar_linear) * np.exp(1j * (phi_phase - np.pi/2))
                
                elif polarization.lower() in ['x', 'l3x']:
                    # For X polarization
                    e_theta[freq_idx, theta_idx, phi_idx] = amplitude * np.cos(np.radians(phi_val)) * np.exp(1j * phi_phase)
                    e_phi[freq_idx, theta_idx, phi_idx] = -amplitude * np.sin(np.radians(phi_val)) * (1/ar_linear) * np.exp(1j * phi_phase)
                
                elif polarization.lower() in ['y', 'l3y']:
                    # For Y polarization
                    e_theta[freq_idx, theta_idx, phi_idx] = amplitude * np.sin(np.radians(phi_val)) * np.exp(1j * phi_phase)
                    e_phi[freq_idx, theta_idx, phi_idx] = amplitude * np.cos(np.radians(phi_val)) * (1/ar_linear) * np.exp(1j * phi_phase)
                
                elif polarization.lower() == 'theta':
                    # For theta polarization
                    e_theta[freq_idx, theta_idx, phi_idx] = amplitude * np.exp(1j * phi_phase)
                    e_phi[freq_idx, theta_idx, phi_idx] = amplitude * (1/ar_linear) * np.exp(1j * phi_phase) * 0.01
                
                elif polarization.lower() == 'phi':
                    # For phi polarization
                    e_theta[freq_idx, theta_idx, phi_idx] = amplitude * (1/ar_linear) * np.exp(1j * phi_phase) * 0.01
                    e_phi[freq_idx, theta_idx, phi_idx] = amplitude * np.exp(1j * phi_phase)
    
    return e_theta, e_phi

def create_gaussian_beam(
    frequencies: np.ndarray,
    theta_angles: np.ndarray,
    phi_angles: np.ndarray,
    edge_taper_db: Union[float, np.ndarray] = -12.0,
    subtended_angle_deg: float = 10.0,
    polarization: str = 'linear',
    peak_gain_dbi: Union[float, np.ndarray] = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Gaussian beam pattern suitable for reflector feed modeling.
    
    Creates a rotationally symmetric Gaussian beam based on edge taper
    at a specified subtended angle.
    
    Args:
        frequencies: Array of frequencies in Hz
        theta_angles: Array of theta angles in degrees
        phi_angles: Array of phi angles in degrees
        edge_taper_db: Edge taper in dB at subtended angle (negative value)
        subtended_angle_deg: Angle from boresight where edge taper is specified
        polarization: 'linear', 'rhcp', or 'lhcp'
        peak_gain_dbi: Peak gain in dBi
        
    Returns:
        Tuple of (e_theta, e_phi) complex arrays with shape [freq, theta, phi]
    """
    # Convert inputs to arrays if needed
    if np.isscalar(peak_gain_dbi):
        peak_gain_dbi = np.full(len(frequencies), peak_gain_dbi)
    if np.isscalar(edge_taper_db):
        edge_taper_db = np.full(len(frequencies), edge_taper_db)
    
    # Initialize arrays
    e_theta = np.zeros((len(frequencies), len(theta_angles), len(phi_angles)), dtype=complex)
    e_phi = np.zeros((len(frequencies), len(theta_angles), len(phi_angles)), dtype=complex)
    
    # Calculate Gaussian parameter from edge taper
    # E(θ) = E_0 * exp(-(θ/θ_0)^2)
    # At subtended angle: edge_taper_dB = 20*log10(exp(-(θ_sub/θ_0)^2))
    # Solving: θ_0 = θ_sub / sqrt(-ln(10^(edge_taper_dB/20)))
    
    for freq_idx, freq in enumerate(frequencies):
        # Calculate Gaussian width parameter
        taper_linear = 10**(edge_taper_db[freq_idx] / 20.0)
        theta_0 = subtended_angle_deg / np.sqrt(-np.log(taper_linear))
        
        # Convert peak gain to field amplitude
        peak_amplitude = np.sqrt(10**(peak_gain_dbi[freq_idx] / 10.0))
        
        # Generate pattern for each angle
        for theta_idx, theta_val in enumerate(theta_angles):
            # Gaussian amplitude pattern (rotationally symmetric)
            amplitude = peak_amplitude * np.exp(-(theta_val / theta_0)**2)
            
            # Add a floor to prevent numerical issues at large angles
            # Typical far sidelobe level for feeds is -40 to -50 dB
            floor_amplitude = peak_amplitude * 10**(-45.0 / 20.0)  # -45 dB floor
            amplitude = np.maximum(amplitude, floor_amplitude)
            
            # Apply to all phi (symmetric)
            if polarization == 'linear':
                e_theta[freq_idx, theta_idx, :] = amplitude
                e_phi[freq_idx, theta_idx, :] = 0.0
            elif polarization == 'rhcp':
                e_theta[freq_idx, theta_idx, :] = amplitude / np.sqrt(2)
                e_phi[freq_idx, theta_idx, :] = -1j * amplitude / np.sqrt(2)
            elif polarization == 'lhcp':
                e_theta[freq_idx, theta_idx, :] = amplitude / np.sqrt(2)
                e_phi[freq_idx, theta_idx, :] = 1j * amplitude / np.sqrt(2)
            else:
                raise ValueError(f"Invalid polarization: {polarization}")
    
    return e_theta, e_phi
