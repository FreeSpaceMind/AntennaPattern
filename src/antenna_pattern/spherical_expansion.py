"""
Spherical wave expansion for antenna patterns.

This module provides functionality to calculate spherical mode coefficients
from far-field antenna patterns using Q-modes as defined in TICRA GRASP.

References:
    Hansen, J.E. (Ed.), "Spherical Near-Field Antenna Measurements", 
    Peter Peregrinus Ltd., London, 1988.
"""

import numpy as np
from scipy.special import lpmv, spherical_jn, spherical_yn
from typing import Tuple, Optional, Dict, Any, Union
import logging

from .utilities import lightspeed, frequency_to_wavelength

logger = logging.getLogger(__name__)


def calculate_mode_index_N(k: float, r0: float) -> int:
    """
    Calculate the maximum polar mode index N for spherical wave expansion.
    
    Based on equation (4.194) from TICRA GRASP manual:
    N = kr₀ + max{3.6√(kr₀), 10}
    
    Args:
        k: Wavenumber (2π/λ) in rad/m
        r0: Radius of minimum sphere in meters
        
    Returns:
        Maximum polar mode index N
    """
    kr0 = k * r0
    N = int(np.floor(kr0 + max(3.6 * np.sqrt(kr0), 10)))
    return N


def normalized_associated_legendre(n: int, m: int, theta: np.ndarray) -> np.ndarray:
    """
    Calculate normalized associated Legendre polynomial.
    
    From equation (4.218):
    P̄ₙᵐ(cos θ) = √[(2n+1)(n-m)! / (2(n+m)!)] Pₙᵐ(cos θ)
    
    Args:
        n: Polar index
        m: Azimuthal index (absolute value)
        theta: Theta angles in radians
        
    Returns:
        Normalized associated Legendre polynomial values
    """
    cos_theta = np.cos(theta)
    
    # Calculate the standard associated Legendre polynomial
    # scipy's lpmv uses (m, n, x) ordering
    Pnm = lpmv(abs(m), n, cos_theta)
    
    # Calculate normalization factor
    from scipy.special import factorial
    norm = np.sqrt((2*n + 1) * factorial(n - abs(m)) / (2 * factorial(n + abs(m))))
    
    return norm * Pnm


def normalized_legendre_derivative(n: int, m: int, theta: np.ndarray) -> np.ndarray:
    """
    Calculate derivative of normalized associated Legendre polynomial with respect to θ.
    
    dP̄ₙᵐ(cos θ)/dθ
    
    Args:
        n: Polar index
        m: Azimuthal index
        theta: Theta angles in radians
        
    Returns:
        Derivative values
    """
    # Use finite difference for now - can be optimized later with analytical formula
    dtheta = 1e-6
    theta_plus = theta + dtheta
    theta_minus = theta - dtheta
    
    P_plus = normalized_associated_legendre(n, m, theta_plus)
    P_minus = normalized_associated_legendre(n, m, theta_minus)
    
    return (P_plus - P_minus) / (2 * dtheta)


def spherical_hankel_second_kind(n: int, kr: np.ndarray) -> np.ndarray:
    """
    Calculate spherical Hankel function of the second kind.
    
    From equation (4.206):
    hₙ⁽²⁾(kr) = j * e^(-jkr) / (kr)  for n=0
    
    For general n, uses the recurrence relation (4.205).
    
    Args:
        n: Order
        kr: Argument (k*r)
        
    Returns:
        Spherical Hankel function values
    """
    # Spherical Hankel function of second kind: h_n^(2)(x) = j_n(x) - i*y_n(x)
    jn = spherical_jn(n, kr)
    yn = spherical_yn(n, kr)
    
    return jn - 1j * yn


def spherical_hankel_derivative(n: int, kr: np.ndarray) -> np.ndarray:
    """
    Calculate derivative of spherical Hankel function.
    
    From recurrence relation (4.208):
    d/d(kr){kr*hₙ⁽²⁾(kr)} = hₙ₋₁⁽²⁾(kr) - n*hₙ⁽²⁾(kr)/(kr)
    
    Args:
        n: Order
        kr: Argument
        
    Returns:
        Derivative values
    """
    if n == 0:
        # Special case for n=0
        h0 = spherical_hankel_second_kind(0, kr)
        h1 = spherical_hankel_second_kind(1, kr)
        return h1 - h0 / kr
    else:
        hn_minus_1 = spherical_hankel_second_kind(n-1, kr)
        hn = spherical_hankel_second_kind(n, kr)
        return hn_minus_1 - n * hn / kr


def calculate_vector_expansion_functions(
    s: int, m: int, n: int, r: Union[float, np.ndarray], 
    theta: Union[float, np.ndarray], phi: Union[float, np.ndarray], k: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate spherical vector expansion functions F_smn(r,θ,φ).
    
    From equations (4.214) and (4.215):
    F₁mn(r,θ,φ) = 1/√(2π√(n(n+1))) (-m/|m|)^m m'mn(r,θ,φ)
    F₂mn(r,θ,φ) = 1/√(2π√(n(n+1))) (-m/|m|)^m n'mn(r,θ,φ)
    
    Args:
        s: Mode type (1 or 2)
        m: Azimuthal index
        n: Polar index
        r: Radius in meters (scalar or array)
        theta: Theta angles in radians (scalar or array)
        phi: Phi angles in radians (scalar or array)
        k: Wavenumber in rad/m
        
    Returns:
        Tuple of (F_r, F_theta, F_phi) components
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    # Broadcast to common shape
    r_grid, theta_grid, phi_grid = np.broadcast_arrays(r, theta, phi)
    
    kr = k * r_grid
    
    # Calculate normalized associated Legendre polynomial and derivative
    Pnm = normalized_associated_legendre(n, m, theta_grid)
    dPnm_dtheta = normalized_legendre_derivative(n, m, theta_grid)
    
    # Calculate spherical Hankel function and derivative
    hn = spherical_hankel_second_kind(n, kr)
    dhn_dkr = spherical_hankel_derivative(n, kr)
    
    # Phase factor (-m/|m|)^m
    if m == 0:
        phase_factor = 1.0
    else:
        phase_factor = (-1.0) ** m
    
    # Normalization
    norm = 1.0 / np.sqrt(2 * np.pi * np.sqrt(n * (n + 1)))
    
    # Azimuthal phase
    exp_imphi = np.exp(1j * m * phi_grid)
    
    # Avoid division by zero in sin(theta)
    sin_theta = np.sin(theta_grid)
    sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    
    # Calculate m'mn or n'mn components based on s
    if s == 1:
        # From equation (4.216): m'mn components
        F_r = np.zeros_like(r_grid, dtype=complex)  # m'mn has no radial component
        
        F_theta = -hn * (dPnm_dtheta / sin_theta_safe) * exp_imphi
        
        F_phi = -hn * (1j * m * Pnm / sin_theta_safe) * exp_imphi
        
    elif s == 2:
        # From equation (4.217): n'mn components  
        F_r = (n * (n + 1) / kr) * hn * Pnm * exp_imphi
        
        F_theta = (1 / kr) * dhn_dkr * dPnm_dtheta * exp_imphi
        
        F_phi = (1j * m / (kr * sin_theta_safe)) * dhn_dkr * Pnm * exp_imphi
    else:
        raise ValueError(f"s must be 1 or 2, got {s}")
    
    # Apply normalization and phase factor
    F_r = norm * phase_factor * F_r
    F_theta = norm * phase_factor * F_theta  
    F_phi = norm * phase_factor * F_phi
    
    return F_r, F_theta, F_phi

def calculate_q_coefficients(
    pattern_obj,
    radius: float,
    frequency_index: int = 0,
    N_theta: Optional[int] = None,
    N_phi: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate spherical wave expansion Q-mode coefficients from far-field pattern.
    
    This function computes the coefficients Q_smn by numerical integration of the
    far-field pattern over a spherical surface. The integration uses the orthogonality
    of spherical modes.
    
    Args:
        pattern_obj: AntennaPattern object containing far-field data
        radius: Radius of the sphere for expansion (minimum sphere radius r₀)
        frequency_index: Index of frequency to use from pattern
        N_theta: Number of theta sample points (if None, determined from pattern)
        N_phi: Number of phi sample points (if None, determined from pattern)
        
    Returns:
        Dictionary containing:
            - 'Q_coefficients': Complex array of shape (2, M, N_modes) where
                                first index is s-1 (s=1,2)
            - 'M': Maximum azimuthal mode index
            - 'N': Maximum polar mode index  
            - 'frequency': Frequency in Hz
            - 'radius': Sphere radius in meters
            - 'mode_power': Total power in modes (for validation)
            
    Notes:
        The Q-coefficients are calculated using:
        Q_smn = ∫∫ E(θ,φ) · F*_smn(θ,φ) sin(θ) dθ dφ
        
        where E is the electric field pattern and F_smn are the vector expansion functions.
    """
    from .spherical_expansion import (
        calculate_mode_index_N,
        calculate_vector_expansion_functions
    )
    
    # Get pattern data for the selected frequency
    freq = pattern_obj.frequencies[frequency_index]
    theta_pattern = pattern_obj.theta_angles  # in degrees
    phi_pattern = pattern_obj.phi_angles  # in degrees
    
    # Get field components
    e_theta = pattern_obj.data.e_theta.values[frequency_index, :, :]
    e_phi = pattern_obj.data.e_phi.values[frequency_index, :, :]
    
    # Convert angles to radians for calculations
    theta_rad = np.radians(theta_pattern)
    phi_rad = np.radians(phi_pattern)
    
    # Calculate wavenumber and wavelength
    wavelength = frequency_to_wavelength(freq)
    k = 2 * np.pi / wavelength
    
    # Determine maximum mode index N
    N = calculate_mode_index_N(k, radius)
    logger.info(f"Maximum polar mode index N = {N} for kr₀ = {k*radius:.2f}")
    
    # Determine M (maximum azimuthal mode index)
    # For far-field patterns, M is typically limited by the pattern symmetry
    # Start with M = N as maximum, can be reduced based on pattern
    M = N
    
    # Check sampling requirements from equations (5), (6), (7), (8)
    dtheta_required = 180.0 / N  # degrees
    dphi_required = 180.0 / (M + 1) if M < N else 360.0 / (2 * N)  # degrees
    
    # Get actual spacing
    if len(theta_pattern) > 1:
        dtheta_actual = np.mean(np.diff(theta_pattern))
    else:
        dtheta_actual = 180.0
        
    if len(phi_pattern) > 1:
        dphi_actual = np.mean(np.diff(phi_pattern))
    else:
        dphi_actual = 360.0
    
    logger.info(f"Sampling check:")
    logger.info(f"  Required: Δθ ≤ {dtheta_required:.2f}°, Δφ ≤ {dphi_required:.2f}°")
    logger.info(f"  Actual:   Δθ = {dtheta_actual:.2f}°, Δφ = {dphi_actual:.2f}°")
    
    if dtheta_actual > dtheta_required or dphi_actual > dphi_required:
        logger.warning("Pattern sampling may be insufficient for accurate mode calculation")
    
    # Initialize coefficient storage
    # Q_coefficients[s-1, m+M, n] for s=1,2; m=-M..M; n=1..N
    Q_coefficients = np.zeros((2, 2*M + 1, N), dtype=complex)
    
    # Create meshgrid for integration
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')
    
    # Integration weight (sin(θ) for solid angle element)
    sin_theta = np.sin(THETA)
    
    # Calculate coefficients for each mode
    # From equation (4.212), we need to find Q_smn such that:
    # E = k√(2ζ) Σ Σ Σ Q_smn F_smn
    
    # Use orthogonality: Q_smn = (1/(k√(2ζ))) ∫∫ E · F*_smn sin(θ) dθ dφ
    
    # Free space impedance
    zeta = 376.730313668  # Ohms
    norm_factor = 1.0 / (k * np.sqrt(2 * zeta))
    
    logger.info(f"Calculating Q-coefficients for {2*(2*M+1)*N} modes...")
    
    for s in [1, 2]:
        for n in range(1, N + 1):
            # For each n, m ranges from -n to n (but limited by M)
            m_min = max(-n, -M)
            m_max = min(n, M)
            
            for m in range(m_min, m_max + 1):
                # Calculate vector expansion function F_smn at all pattern points
                # Note: In far field, we evaluate at large radius
                F_r, F_theta, F_phi = calculate_vector_expansion_functions(
                    s, m, n, radius, THETA, PHI, k
                )
                
                # Calculate inner product E · F*
                # In spherical coordinates: E · F* = E_θ F*_θ + E_φ F*_φ
                # (radial component is zero in far field)
                integrand = (e_theta * np.conj(F_theta) + 
                            e_phi * np.conj(F_phi)) * sin_theta
                
                # Numerical integration using trapezoidal rule
                # ∫∫ f(θ,φ) sin(θ) dθ dφ
                Q_smn = np.trapz(np.trapz(integrand, phi_rad, axis=1), theta_rad, axis=0)
                
                # Apply normalization
                Q_smn *= norm_factor
                
                # Store coefficient
                # Index: Q_coefficients[s-1, m+M, n-1]
                Q_coefficients[s-1, m + M, n-1] = Q_smn
        
        logger.info(f"  Completed s={s} modes")
    
    # Calculate total power in modes (equation 4.219)
    total_power = 0.0
    for s in [1, 2]:
        for n in range(1, N + 1):
            for m in range(-n, n + 1):
                if abs(m) <= M:
                    Q_smn = Q_coefficients[s-1, m + M, n-1]
                    # Power contribution: n(n+1)(n+|m|)! / (2n+1)(n-|m|)! * |Q_smn|²
                    from scipy.special import factorial
                    power_factor = (n * (n + 1) * factorial(n + abs(m)) / 
                                  ((2*n + 1) * factorial(n - abs(m))))
                    total_power += power_factor * abs(Q_smn)**2
    
    total_power *= 0.5  # Factor from equation 4.219
    
    logger.info(f"Total power in modes: {total_power:.6e} W")
    
    return {
        'Q_coefficients': Q_coefficients,
        'M': M,
        'N': N,
        'frequency': freq,
        'wavelength': wavelength,
        'radius': radius,
        'mode_power': total_power,
        'k': k
    }


def add_swe_to_pattern(pattern_obj, swe_data: Dict[str, Any]) -> None:
    """
    Add spherical wave expansion data to an AntennaPattern object.
    
    Args:
        pattern_obj: AntennaPattern to add SWE data to
        swe_data: Dictionary returned by calculate_q_coefficients
    """
    if not hasattr(pattern_obj, 'swe'):
        pattern_obj.swe = {}
    
    # Store SWE data indexed by frequency
    freq = swe_data['frequency']
    pattern_obj.swe[freq] = swe_data
    
    logger.info(f"Added SWE coefficients to pattern for f={freq/1e9:.3f} GHz")

def evaluate_field_from_modes(
    Q_coefficients: np.ndarray,
    M: int,
    N: int,
    k: float,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate electric field from spherical mode coefficients at specified points.
    
    Evaluates equation (4.199):
    E(r,θ,φ) = k√(2ζ) Σ Σ Σ Q_smn F_smn(r,θ,φ)
    
    Args:
        Q_coefficients: Mode coefficients array [s, m, n] where:
                       s index: 0,1 for s=1,2
                       m index: 0..2M for m=-M..M
                       n index: 0..N-1 for n=1..N
        M: Maximum azimuthal mode index
        N: Maximum polar mode index
        k: Wavenumber (2π/λ) in rad/m
        r: Radial distances in meters (scalar or array)
        theta: Theta angles in radians (scalar or array)
        phi: Phi angles in radians (scalar or array)
        
    Returns:
        Tuple of (E_r, E_theta, E_phi) field components in V/m
        
    Notes:
        - All arrays must be broadcastable to the same shape
        - For spherical surface: r is scalar, theta/phi are arrays
        - For arbitrary points: all can be arrays of same shape
    """
    from .spherical_expansion import calculate_vector_expansion_functions
    
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    # Broadcast to common shape
    r_grid, theta_grid, phi_grid = np.broadcast_arrays(r, theta, phi)
    output_shape = r_grid.shape
    
    # Initialize field components
    E_r = np.zeros(output_shape, dtype=complex)
    E_theta = np.zeros(output_shape, dtype=complex)
    E_phi = np.zeros(output_shape, dtype=complex)
    
    # Free space impedance
    zeta = 376.730313668  # Ohms
    
    # Normalization factor from equation 4.199
    norm = k * np.sqrt(2 * zeta)
    
    # Sum over all modes
    logger.info(f"Evaluating field at {r_grid.size} points using {2*(2*M+1)*N} modes...")
    
    for s in [1, 2]:
        s_idx = s - 1
        
        for n in range(1, N + 1):
            n_idx = n - 1
            
            # For this n, m ranges from -n to n (limited by M)
            m_min = max(-n, -M)
            m_max = min(n, M)
            
            for m in range(m_min, m_max + 1):
                m_idx = m + M
                
                # Get coefficient
                Q_smn = Q_coefficients[s_idx, m_idx, n_idx]
                
                # Skip if coefficient is negligible
                if abs(Q_smn) < 1e-15:
                    continue
                
                # Calculate vector expansion function at all points
                # This is the most expensive operation
                F_r, F_theta, F_phi = calculate_vector_expansion_functions(
                    s, m, n, r_grid, theta_grid, phi_grid, k
                )
                
                # Add contribution from this mode
                E_r += norm * Q_smn * F_r
                E_theta += norm * Q_smn * F_theta
                E_phi += norm * Q_smn * F_phi
    
    logger.info("Field evaluation complete")
    
    return E_r, E_theta, E_phi


def calculate_nearfield_spherical_surface(
    swe_data: Dict[str, Any],
    radius: float,
    theta_points: np.ndarray,
    phi_points: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate near-field on a spherical surface from SWE coefficients.
    
    Args:
        swe_data: Dictionary from calculate_spherical_modes containing coefficients
        radius: Radius of evaluation sphere in meters
        theta_points: Theta angles in degrees for evaluation
        phi_points: Phi angles in degrees for evaluation
        
    Returns:
        Dictionary containing:
            - 'E_r': Radial electric field component [theta, phi]
            - 'E_theta': Theta electric field component [theta, phi]
            - 'E_phi': Phi electric field component [theta, phi]
            - 'radius': Evaluation radius
            - 'theta': Theta angles in degrees
            - 'phi': Phi angles in degrees
            - 'frequency': Frequency in Hz
            
    Example:
        ```python
        # Calculate modes
        swe_data = pattern.calculate_spherical_modes()
        
        # Evaluate on spherical surface at 10 cm radius
        theta = np.linspace(-90, 90, 181)
        phi = np.linspace(0, 360, 37)
        nearfield = calculate_nearfield_spherical_surface(
            swe_data, radius=0.10, theta_points=theta, phi_points=phi
        )
        
        # Access field components
        E_theta = nearfield['E_theta']
        E_phi = nearfield['E_phi']
        ```
    """
    # Extract data from swe_data
    Q_coefficients = swe_data['Q_coefficients']
    M = swe_data['M']
    N = swe_data['N']
    k = swe_data['k']
    
    # Convert angles to radians
    theta_rad = np.radians(theta_points)
    phi_rad = np.radians(phi_points)
    
    # Create meshgrid for evaluation
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')
    
    # Evaluate field - r is scalar (broadcast to grid shape)
    E_r, E_theta, E_phi = evaluate_field_from_modes(
        Q_coefficients, M, N, k,
        radius,  # Scalar radius
        THETA,   # 2D array
        PHI      # 2D array
    )
    
    return {
        'E_r': E_r,
        'E_theta': E_theta,
        'E_phi': E_phi,
        'radius': radius,
        'theta': theta_points,
        'phi': phi_points,
        'frequency': swe_data['frequency'],
        'wavelength': swe_data['wavelength']
    }


def calculate_nearfield_planar_surface(
    swe_data: Dict[str, Any],
    x_points: np.ndarray,
    y_points: np.ndarray,
    z_plane: float
) -> Dict[str, Any]:
    """
    Calculate near-field on a planar surface from SWE coefficients.
    
    The plane is defined at z=z_plane with x,y coordinates.
    
    Args:
        swe_data: Dictionary from calculate_spherical_modes containing coefficients
        x_points: X coordinates in meters for evaluation
        y_points: Y coordinates in meters for evaluation
        z_plane: Z coordinate of plane in meters
        
    Returns:
        Dictionary containing:
            - 'E_x': X electric field component [x, y]
            - 'E_y': Y electric field component [x, y]
            - 'E_z': Z electric field component [x, y]
            - 'x': X coordinates in meters
            - 'y': Y coordinates in meters
            - 'z': Z plane coordinate in meters
            - 'frequency': Frequency in Hz
            
    Notes:
        - Converts cartesian to spherical coordinates for each point
        - Then transforms spherical field components back to cartesian
        
    Example:
        ```python
        # Calculate modes
        swe_data = pattern.calculate_spherical_modes()
        
        # Evaluate on planar surface
        x = np.linspace(-0.1, 0.1, 51)  # ±10 cm
        y = np.linspace(-0.1, 0.1, 51)
        nearfield = calculate_nearfield_planar_surface(
            swe_data, x_points=x, y_points=y, z_plane=0.05
        )
        
        # Access field components
        E_x = nearfield['E_x']
        E_y = nearfield['E_y']
        ```
    """
    # Extract data from swe_data
    Q_coefficients = swe_data['Q_coefficients']
    M = swe_data['M']
    N = swe_data['N']
    k = swe_data['k']
    
    # Create meshgrid for x,y coordinates
    X, Y = np.meshgrid(x_points, y_points, indexing='ij')
    Z = np.full_like(X, z_plane)
    
    # Convert cartesian to spherical coordinates
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(Z / r)  # Angle from z-axis
    phi = np.arctan2(Y, X)    # Angle in x-y plane
    
    # Evaluate field in spherical coordinates
    E_r, E_theta, E_phi = evaluate_field_from_modes(
        Q_coefficients, M, N, k,
        r, theta, phi
    )
    
    # Convert field from spherical to cartesian coordinates
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # Transformation matrix from spherical to cartesian
    # [E_x]   [sin(θ)cos(φ)  cos(θ)cos(φ)  -sin(φ)] [E_r    ]
    # [E_y] = [sin(θ)sin(φ)  cos(θ)sin(φ)   cos(φ)] [E_theta]
    # [E_z]   [cos(θ)        -sin(θ)          0    ] [E_phi  ]
    
    E_x = (sin_theta * cos_phi * E_r + 
           cos_theta * cos_phi * E_theta - 
           sin_phi * E_phi)
    
    E_y = (sin_theta * sin_phi * E_r + 
           cos_theta * sin_phi * E_theta + 
           cos_phi * E_phi)
    
    E_z = (cos_theta * E_r - 
           sin_theta * E_theta)
    
    return {
        'E_x': E_x,
        'E_y': E_y,
        'E_z': E_z,
        'x': x_points,
        'y': y_points,
        'z': z_plane,
        'frequency': swe_data['frequency'],
        'wavelength': swe_data['wavelength']
    }

def calculate_mode_power_distribution(Q_coefficients: np.ndarray, M: int, N: int) -> np.ndarray:
    """
    Calculate power contribution of each polar mode index n.
    
    From equation (4.219), the total power is:
    P = (1/2) Σ_{s=1,2} Σ_{n=1}^N Σ_{m=-n}^n |Q_smn|²
    
    This function returns power per n value for analysis.
    
    Args:
        Q_coefficients: Array of shape (2, 2M+1, N)
        M: Maximum azimuthal index
        N: Maximum polar index
        
    Returns:
        Array of length N containing power in each polar mode
    """
    power_per_n = np.zeros(N)
    
    for n in range(1, N + 1):
        power_n = 0.0
        for s in [0, 1]:  # s-1 = 0,1 for s=1,2
            for m in range(-n, n + 1):
                if abs(m) <= M:
                    Q_smn = Q_coefficients[s, m + M, n - 1]
                    power_n += abs(Q_smn)**2
        
        power_per_n[n - 1] = 0.5 * power_n
    
    return power_per_n


def find_truncation_index(power_per_n: np.ndarray, power_threshold: float = 0.99) -> int:
    """
    Find the mode index where cumulative power reaches the threshold.
    
    Args:
        power_per_n: Power in each mode
        power_threshold: Fraction of total power to capture (default 0.99)
        
    Returns:
        Mode index n where cumulative power exceeds threshold
    """
    total_power = np.sum(power_per_n)
    cumulative_power = np.cumsum(power_per_n)
    cumulative_fraction = cumulative_power / total_power
    
    # Find first index where we exceed threshold
    indices = np.where(cumulative_fraction >= power_threshold)[0]
    
    if len(indices) == 0:
        # Threshold not reached, return max index
        return len(power_per_n)
    else:
        # Return n value (1-indexed), not array index
        return indices[0] + 1


def check_convergence(power_per_n: np.ndarray, N: int, 
                     convergence_threshold: float = 0.01) -> Tuple[bool, float]:
    """
    Check if power in highest modes is below threshold (convergence achieved).
    
    Args:
        power_per_n: Power in each mode
        N: Maximum mode index
        convergence_threshold: Fraction of total power allowed in highest modes
        
    Returns:
        Tuple of (converged, fraction_in_high_modes)
    """
    total_power = np.sum(power_per_n)
    
    # Check power in highest 10% of modes (or at least last 5 modes)
    n_check = max(5, int(0.1 * N))
    high_mode_power = np.sum(power_per_n[-n_check:])
    
    fraction = high_mode_power / total_power if total_power > 0 else 0.0
    converged = fraction < convergence_threshold
    
    return converged, fraction


def calculate_q_coefficients_adaptive(
    pattern_obj,
    initial_radius: Optional[float] = None,
    frequency_index: int = 0,
    power_threshold: float = 0.99,
    convergence_threshold: float = 0.01,
    max_iterations: int = 10,
    radius_growth_factor: float = 1.5
) -> Dict[str, Any]:
    """
    Calculate spherical wave expansion Q-mode coefficients with adaptive radius selection.
    
    This function automatically determines the optimal minimum sphere radius by:
    1. Starting with an initial guess (default: 5 wavelengths)
    2. Computing Q-coefficients and checking power distribution
    3. If significant power remains in highest modes, increasing radius
    4. Once converged, truncating modes that contribute < 1% of total power
    
    Args:
        pattern_obj: AntennaPattern object containing far-field data
        initial_radius: Initial guess for minimum sphere radius in meters
                       If None, uses 5 wavelengths
        frequency_index: Index of frequency to use from pattern
        power_threshold: Fraction of power to retain in final modes (default 0.99)
        convergence_threshold: Max fraction of power allowed in highest modes (default 0.01)
        max_iterations: Maximum number of radius increase iterations
        radius_growth_factor: Factor to increase radius each iteration (default 1.5)
        
    Returns:
        Dictionary containing:
            - 'Q_coefficients': Truncated complex array [s, m, n]
            - 'M': Maximum azimuthal mode index
            - 'N': Truncated polar mode index (modes containing power_threshold of power)
            - 'N_full': Full polar mode index before truncation
            - 'frequency': Frequency in Hz
            - 'radius': Final sphere radius in meters
            - 'mode_power': Total power in modes
            - 'power_per_n': Power distribution across modes
            - 'converged': Whether calculation converged
            - 'iterations': Number of iterations used
            
    Example:
        ```python
        # Automatically determine radius
        swe_data = pattern.calculate_spherical_modes()
        
        # Will find optimal radius, report final N and total power
        print(f"Final radius: {swe_data['radius']:.4f} m")
        print(f"Modes used: N={swe_data['N']} (truncated from {swe_data['N_full']})")
        ```
    """
    from .spherical_expansion import calculate_q_coefficients
    from .utilities import frequency_to_wavelength
    
    # Get frequency and wavelength
    freq = pattern_obj.frequencies[frequency_index]
    wavelength = frequency_to_wavelength(freq)
    
    # Set initial radius if not provided
    if initial_radius is None:
        radius = 5.0 * wavelength
        logger.info(f"Using initial radius = 5λ = {radius:.4f} m")
    else:
        radius = initial_radius
        logger.info(f"Using initial radius = {radius:.4f} m")
    
    # Iteration loop
    converged = False
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n=== Iteration {iteration}: r₀ = {radius:.4f} m ({radius/wavelength:.2f}λ) ===")
        
        # Calculate coefficients for current radius
        swe_data = calculate_q_coefficients(
            pattern_obj, radius, frequency_index
        )
        
        Q_coefficients = swe_data['Q_coefficients']
        M = swe_data['M']
        N = swe_data['N']
        
        # Analyze power distribution
        power_per_n = calculate_mode_power_distribution(Q_coefficients, M, N)
        
        # Check convergence
        converged, high_mode_fraction = check_convergence(
            power_per_n, N, convergence_threshold
        )
        
        logger.info(f"  N = {N} modes calculated")
        logger.info(f"  Power in highest modes: {high_mode_fraction*100:.2f}%")
        
        if converged:
            logger.info(f"  ✓ Converged! (< {convergence_threshold*100:.1f}% in highest modes)")
            break
        else:
            if iteration < max_iterations:
                radius *= radius_growth_factor
                logger.info(f"  ✗ Not converged, increasing radius by {radius_growth_factor}x")
            else:
                logger.warning(f"  ! Max iterations reached without full convergence")
    
    # Find truncation point (where we have power_threshold of total power)
    N_truncated = find_truncation_index(power_per_n, power_threshold)
    cumulative_power = np.cumsum(power_per_n)
    total_power = cumulative_power[-1]
    power_retained = cumulative_power[N_truncated - 1]
    
    logger.info(f"\n=== Final Results ===")
    logger.info(f"  Final radius: {radius:.4f} m ({radius/wavelength:.2f}λ)")
    logger.info(f"  Full mode count: N = {N}")
    logger.info(f"  Truncated to: N = {N_truncated} (retains {power_retained/total_power*100:.2f}% power)")
    logger.info(f"  Total power: {total_power:.6e} W")
    logger.info(f"  Iterations: {iteration}")
    
    # Truncate coefficient array
    Q_truncated = Q_coefficients[:, :, :N_truncated]
    
    # Also truncate M if possible (modes with |m| > N_truncated are zero)
    M_truncated = min(M, N_truncated)
    m_start = M - M_truncated
    m_end = M + M_truncated + 1
    Q_truncated = Q_truncated[:, m_start:m_end, :]
    
    logger.info(f"  Azimuthal modes: M = {M_truncated} (truncated from {M})")
    
    # Return modified data structure
    result = {
        'Q_coefficients': Q_truncated,
        'M': M_truncated,
        'N': N_truncated,
        'N_full': N,
        'M_full': M,
        'frequency': freq,
        'wavelength': wavelength,
        'radius': radius,
        'mode_power': total_power,
        'power_per_n': power_per_n,
        'k': swe_data['k'],
        'converged': converged,
        'iterations': iteration,
        'power_retained_fraction': power_retained / total_power
    }
    
    return result