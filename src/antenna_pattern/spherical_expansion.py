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
    Calculate maximum polar mode index N.
    
    From Jensen et al. (2004):
    N = kr₀ + 3.6*(kr₀)^(1/3) for -60dB truncation
    N = kr₀ + 5.0*(kr₀)^(1/3) for -80dB truncation
    """
    kr0 = k * r0
    
    # Use -60dB formula (good balance of accuracy and speed)
    N = int(np.floor(kr0 + 3.6 * np.power(kr0, 1.0/3.0)))
    
    return N


def normalized_associated_legendre(n: int, m: int, theta: np.ndarray) -> np.ndarray:
    """
    Calculate normalized associated Legendre polynomial.
    """
    from scipy.special import gammaln
    
    # Critical check: |m| must be <= n
    if abs(m) > n:
        # Return zeros - this mode doesn't exist
        return np.zeros_like(theta)
    
    cos_theta = np.cos(theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    Pnm = lpmv(abs(m), n, cos_theta)
    
    # Calculate normalization factor using log-gamma
    log_norm = 0.5 * (
        np.log(2*n + 1) + 
        gammaln(n - abs(m) + 1) - 
        np.log(2) - 
        gammaln(n + abs(m) + 1)
    )
    norm = np.exp(log_norm)
    
    return norm * Pnm


def normalized_legendre_derivative(n: int, m: int, theta: np.ndarray) -> np.ndarray:
    """
    Calculate derivative of normalized associated Legendre polynomial with respect to θ.
    """
    # Critical check: |m| must be <= n
    if abs(m) > n:
        return np.zeros_like(theta)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Get current Legendre polynomial
    Pnm = normalized_associated_legendre(n, m, theta)
    
    if n == 0:
        return np.zeros_like(theta)
    
    # Check if we can use the recurrence relation
    # We need |m| <= (n-1) to compute P_{n-1}^m
    if abs(m) <= (n - 1):
        Pnm_minus1 = normalized_associated_legendre(n-1, m, theta)
        
        # Avoid division by zero at poles
        sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        
        # dP/dθ = [n*cos(θ)*P_n^m - (n+|m|)*P_{n-1}^m] / sin(θ)
        dPnm = (n * cos_theta * Pnm - (n + abs(m)) * Pnm_minus1) / sin_theta_safe
    else:
        # For |m| = n, use alternative formula or finite difference
        # Since P_n^n has a simple form, its derivative is also simple
        sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        
        # For |m| = n: dP/dθ = n * cos(θ) * P_n^m / sin(θ)
        dPnm = n * cos_theta * Pnm / sin_theta_safe
    
    return dPnm


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
    # Validate inputs
    if abs(m) > n:
        logger.warning(f"Invalid mode: |m|={abs(m)} > n={n}, returning zeros")
        r = np.atleast_1d(r)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        r_grid, theta_grid, phi_grid = np.broadcast_arrays(r, theta, phi)
        return (np.zeros_like(r_grid, dtype=complex), 
                np.zeros_like(r_grid, dtype=complex), 
                np.zeros_like(r_grid, dtype=complex))

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
    logger.info(f"  Required: dtheta <= {dtheta_required:.2f}deg, dphi <= {dphi_required:.2f}deg")
    logger.info(f"  Actual:   dtheta = {dtheta_actual:.2f}deg, dphi = {dphi_actual:.2f}deg")
    
    if dtheta_actual > dtheta_required or dphi_actual > dphi_required:
        logger.warning("Pattern sampling may be insufficient for accurate mode calculation")
    
    # LIMIT N based on pattern sampling to avoid wasted computation
    dtheta_actual_rad = np.radians(dtheta_actual)
    max_N_from_sampling = int(np.pi / dtheta_actual_rad)  # Nyquist limit
    if N > max_N_from_sampling:
        logger.info(f"Limiting N from {N} to {max_N_from_sampling} based on pattern sampling")
        N = max_N_from_sampling
        M = N  # Keep M = N

    # Initialize coefficient storage
    # Q_coefficients[s-1, m+M, n] for s=1,2; m=-M..M; n=1..N
    Q_coefficients = np.zeros((2, 2*M + 1, N), dtype=complex)
    
    # Create meshgrid for integration
    THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')
    sin_theta = np.sin(THETA)
    
    # PRE-COMPUTE ALL LEGENDRE POLYNOMIALS 
    legendre_cache = precompute_legendre_polynomials_vectorized(N, M, THETA)

    # PRE-COMPUTE SPHERICAL HANKEL FUNCTIONS
    logger.info("Pre-computing spherical Hankel functions...")
    kr = k * radius 
    hankel_cache = {}
    for n in range(1, N + 1):
        hankel_cache[n] = spherical_hankel_second_kind(n, kr)
        hankel_cache[(n, 'deriv')] = spherical_hankel_derivative(n, kr)
    logger.info(f"Cached {len(hankel_cache)//2} Hankel function pairs")

    # Initialize coefficient storage
    Q_coefficients = np.zeros((2, 2*M + 1, N), dtype=complex)
    
    # Free space impedance
    zeta = 376.730313668
    norm_factor = 1.0 / (k * np.sqrt(2 * zeta))
    
    logger.info(f"Starting integration for modes...")
    mode_count = 0
    total_modes = sum(min(2*n+1, 2*M+1) for n in range(1, N+1)) * 2
    
    for s in [1, 2]:
        for n in range(1, N + 1):
            m_min = max(-n, -M)
            m_max = min(n, M)
            
            # Get cached Hankel functions (same for all m at this n)
            hn = hankel_cache[n]
            dhn_dkr = hankel_cache[(n, 'deriv')]
            
            for m in range(m_min, m_max + 1):
                mode_count += 1

                if abs(m) > n:
                    continue
                
                # GET CACHED LEGENDRE POLYNOMIALS (no recalculation!)
                Pnm = legendre_cache[(n, abs(m))]
                dPnm_dtheta = legendre_cache[(n, abs(m), 'deriv')]
                
                # Phase factor and normalization
                phase_factor = 1.0 if m == 0 else (-1.0) ** m
                norm = 1.0 / np.sqrt(2 * np.pi * np.sqrt(n * (n + 1)))
                exp_imphi = np.exp(1j * m * PHI)
                
                # Avoid division by zero
                sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
                
                # Calculate F components based on cached values
                if s == 1:
                    F_theta = -hn * (dPnm_dtheta / sin_theta_safe) * exp_imphi
                    F_phi = -hn * (1j * m * Pnm / sin_theta_safe) * exp_imphi
                else:  # s == 2
                    F_r = (n * (n + 1) / kr) * hn * Pnm * exp_imphi
                    F_theta = (1 / kr) * dhn_dkr * dPnm_dtheta * exp_imphi
                    F_phi = (1j * m / (kr * sin_theta_safe)) * dhn_dkr * Pnm * exp_imphi
                
                # Apply normalization and phase
                F_theta = norm * phase_factor * F_theta
                F_phi = norm * phase_factor * F_phi
                
                # Calculate inner product and integrate
                integrand = (e_theta * np.conj(F_theta) + 
                            e_phi * np.conj(F_phi)) * sin_theta
                
                Q_smn = np.trapezoid(np.trapezoid(integrand, phi_rad, axis=1), theta_rad, axis=0)
                Q_smn *= norm_factor
                
                Q_coefficients[s-1, m + M, n-1] = Q_smn

                if s == 1 and m == 0 and n <= 5:
                    logger.info(f"    Q_{s},{m},{n} = {abs(Q_smn):.6e}")
        
        logger.info(f"  Completed s={s} modes")
    
    # Calculate total power in modes (equation 4.219)
    total_power = 0.0
    for s in [1, 2]:
        for n in range(1, N + 1):
            for m in range(-n, n + 1):
                if abs(m) <= M:
                    Q_smn = Q_coefficients[s-1, m + M, n-1]
                    # REMOVED FACTORIAL - it causes overflow and is wrong!
                    # Just use |Q_smn|² 
                    total_power += abs(Q_smn)**2

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
    Calculate power per mode n.
    
    From Jensen paper Eq. (7): P_rad(n) = Σ_s,m |Q_smn|²
    NO factorial weighting!
    """
    power_per_n = np.zeros(N)
    
    for n in range(1, N + 1):
        power_n = 0.0
        for s in [0, 1]:  # s=1,2 → index 0,1
            m_min = max(-n, -M)
            m_max = min(n, M)
            for m in range(m_min, m_max + 1):
                Q_smn = Q_coefficients[s, m + M, n - 1]
                power_n += abs(Q_smn)**2  # Simple |Q|² sum
        
        power_per_n[n - 1] = power_n
    
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
                     convergence_threshold: float = 0.10) -> Tuple[bool, float]:
    """
    Check if power in highest modes is below threshold.
    
    CRITICAL: power_per_n[0] is n=1 (low mode), power_per_n[-1] is n=N (high mode)
    """
    total_power = np.sum(power_per_n)
    
    if total_power == 0 or not np.isfinite(total_power):
        logger.error(f"Invalid total power: {total_power}")
        return False, 1.0
    
    # Check power in highest 10% of modes
    n_check = max(5, int(0.1 * N))
    
    # FIXED: Check the LAST n_check modes (highest n)
    high_mode_power = np.sum(power_per_n[-n_check:])
    
    # Also check FIRST modes (should have MOST power for directive antenna)
    low_mode_power = np.sum(power_per_n[:10])
    
    fraction_high = high_mode_power / total_power
    fraction_low = low_mode_power / total_power
    
    logger.info(f"  Power check: first 10 modes={fraction_low*100:.1f}%, last {n_check} modes={fraction_high*100:.1f}%")
    
    # Sanity check: if most power is in first modes, that's GOOD
    if fraction_low > 0.5:
        logger.info(f"  Good power distribution (most power in low modes)")
        converged = True
    else:
        converged = fraction_high < convergence_threshold
    
    return converged, fraction_high

def calculate_q_coefficients_adaptive(
    pattern_obj,
    initial_radius: Optional[float] = None,
    frequency_index: int = 0,
    power_threshold: float = 0.99,
    convergence_threshold: float = 0.10,  # Relaxed - 10% in high modes is ok
    max_iterations: int = 3,  # Reduced - should rarely need more than 2
    radius_growth_factor: float = 1.5
) -> Dict[str, Any]:
    """
    Calculate spherical wave expansion Q-mode coefficients with smart radius selection.
    
    This function automatically determines the optimal minimum sphere radius by:
    1. Estimating antenna size from far-field pattern beamwidth
    2. Using conservative initial radius (should work on first try)
    3. Validating power distribution
    4. Only re-calculating if validation fails (rare)
    5. Truncating modes to retain power_threshold of total power
    
    Args:
        pattern_obj: AntennaPattern object containing far-field data
        initial_radius: Override for minimum sphere radius in meters
                       If None, estimates from pattern
        frequency_index: Index of frequency to use from pattern
        power_threshold: Fraction of power to retain in final modes (default 0.99)
        convergence_threshold: Max fraction of power allowed in highest modes (default 0.10)
        max_iterations: Maximum iterations if refinement needed (default 3)
        radius_growth_factor: Factor to increase radius if needed (default 1.5)
        
    Returns:
        Dictionary containing truncated Q-coefficients and metadata
    """
    from .utilities import frequency_to_wavelength
    
    # Get frequency and wavelength
    freq = pattern_obj.frequencies[frequency_index]
    wavelength = frequency_to_wavelength(freq)
    
    # SMART INITIAL RADIUS GUESS
    if initial_radius is None:
        radius = estimate_antenna_radius_from_pattern(pattern_obj, frequency_index)
    else:
        radius = initial_radius
        logger.info(f"Using specified initial radius: {radius:.4f} m ({radius/wavelength:.1f} lambda)")
    
    # Iteration loop (should converge in 1-2 iterations)
    converged = False
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n=== Iteration {iteration}: r0 = {radius:.4f} m ({radius/wavelength:.1f} lambda) ===")
        
        # Calculate coefficients for current radius
        swe_data = calculate_q_coefficients(
            pattern_obj, radius, frequency_index
        )
        
        Q_coefficients = swe_data['Q_coefficients']
        M = swe_data['M']
        N = swe_data['N']
        
        # Analyze power distribution
        power_per_n = calculate_mode_power_distribution(Q_coefficients, M, N)
        
        # VALIDATE: Check if power distribution is reasonable
        converged, high_mode_fraction = check_convergence(
            power_per_n, N, convergence_threshold
        )
        
        logger.info(f"  N = {N} modes calculated")
        logger.info(f"  Power in highest modes: {high_mode_fraction*100:.1f}%")
        
        if converged:
            logger.info(f"  OK - Converged! Power distribution is reasonable.")
            break
        else:
            if iteration < max_iterations:
                radius *= radius_growth_factor
                logger.info(f"  X Not converged, increasing radius by {radius_growth_factor}x")
            else:
                logger.warning(f"  ! Max iterations reached - using current result")
                logger.warning(f"    High-mode power: {high_mode_fraction*100:.1f}% (threshold: {convergence_threshold*100:.1f}%)")
    
    # Truncate modes to power threshold
    N_truncated = find_truncation_index(power_per_n, power_threshold)
    cumulative_power = np.cumsum(power_per_n)
    total_power = cumulative_power[-1]
    power_retained = cumulative_power[N_truncated - 1]
    
    logger.info(f"\n=== Final Results ===")
    logger.info(f"  Final radius: {radius:.4f} m ({radius/wavelength:.1f} lambda)")
    logger.info(f"  Full mode count: N = {N}")
    logger.info(f"  Truncated to: N = {N_truncated} (retains {power_retained/total_power*100:.1f}% power)")
    logger.info(f"  Total power: {total_power:.6e} W")
    logger.info(f"  Iterations: {iteration}")
    
    # Truncate coefficient array
    Q_truncated = Q_coefficients[:, :, :N_truncated]
    
    # Also truncate M if possible
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

def precompute_legendre_polynomials_vectorized(N: int, M: int, theta: np.ndarray) -> Dict:
    """
    Vectorized computation of all Legendre polynomials and derivatives.
    
    Much faster than computing each (n,m) pair individually.
    
    Args:
        N: Maximum polar mode index
        M: Maximum azimuthal mode index  
        theta: Theta angles in radians (2D grid)
        
    Returns:
        Dictionary with keys (n, m) and (n, m, 'deriv')
    """
    from scipy.special import lpmv, gammaln
    
    logger.info(f"Vectorized pre-computation of Legendre polynomials for N={N}, M={M}...")
    
    cos_theta = np.cos(theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sin(theta)
    sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    
    cache = {}
    
    # For each m, compute all n values at once (vectorized over n)
    for m in range(M + 1):
        # Create array of n values from max(1, m) to N
        n_start = max(1, m)
        n_values = np.arange(n_start, N + 1)
        
        if len(n_values) == 0:
            continue
        
        # Compute Pnm for all n at once - this is the key speedup!
        # Shape: (len(n_values), theta.shape[0], theta.shape[1])
        Pnm_array = np.zeros((len(n_values),) + theta.shape)
        
        for i, n in enumerate(n_values):
            # Standard Legendre polynomial
            Pnm = lpmv(m, n, cos_theta)
            
            # Normalization factor using log-gamma
            log_norm = 0.5 * (
                np.log(2*n + 1) + 
                gammaln(n - m + 1) - 
                np.log(2) - 
                gammaln(n + m + 1)
            )
            norm = np.exp(log_norm)
            
            Pnm_array[i] = norm * Pnm
        
        # Compute derivatives using recurrence relation
        # dP/dθ for all n at once
        dPnm_array = np.zeros_like(Pnm_array)
        
        for i, n in enumerate(n_values):
            if n == m:
                # For n = m, use simple formula
                dPnm_array[i] = n * cos_theta * Pnm_array[i] / sin_theta_safe
            else:
                # Use recurrence: dP_n^m/dθ = [n*cos(θ)*P_n^m - (n+m)*P_{n-1}^m] / sin(θ)
                if i > 0:  # We have P_{n-1}^m available
                    dPnm_array[i] = (n * cos_theta * Pnm_array[i] - 
                                     (n + m) * Pnm_array[i-1]) / sin_theta_safe
                else:
                    # n-1 < m, so we need to get it separately
                    if n > 1:
                        Pnm_minus1 = normalized_associated_legendre(n-1, m, theta)
                        dPnm_array[i] = (n * cos_theta * Pnm_array[i] - 
                                        (n + m) * Pnm_minus1) / sin_theta_safe
                    else:
                        dPnm_array[i] = n * cos_theta * Pnm_array[i] / sin_theta_safe
        
        # Store in cache
        for i, n in enumerate(n_values):
            cache[(n, m)] = Pnm_array[i]
            cache[(n, m, 'deriv')] = dPnm_array[i]
    
    logger.info(f"Cached {len(cache)//2} Legendre polynomial pairs")
    return cache

def estimate_antenna_radius_from_pattern(pattern_obj, frequency_index: int = 0) -> float:
    """
    Estimate antenna radius including aperture AND depth.
    
    For horns: minimum sphere must enclose entire horn structure.
    """
    from .utilities import frequency_to_wavelength
    
    freq = pattern_obj.frequencies[frequency_index]
    wavelength = frequency_to_wavelength(freq)
    
    try:
        theta = pattern_obj.theta_angles
        e_co = pattern_obj.data.e_co.values[frequency_index, :, 0]
        
        max_val = np.max(np.abs(e_co))
        threshold = max_val / np.sqrt(2)
        above_threshold = np.abs(e_co) >= threshold
        
        if np.sum(above_threshold) > 1:
            theta_indices = np.where(above_threshold)[0]
            theta_3dB_deg = theta[theta_indices[-1]] - theta[theta_indices[0]]
            theta_3dB_rad = np.radians(theta_3dB_deg)
            
            if theta_3dB_rad > 0:
                # Estimate aperture diameter
                D_aperture = 1.2 * wavelength / theta_3dB_rad
                
                # For horns: add depth estimate (typically 2-3× aperture diameter)
                # Conservative: use 3× diameter for sphere radius
                radius = D_aperture * 1.5  # Total radius = 1.5× diameter
                
                logger.info(f"Estimated antenna radius from beamwidth ({theta_3dB_deg:.1f} deg):")
                logger.info(f"  Aperture diameter: {D_aperture:.4f} m ({D_aperture/wavelength:.1f} lambda)")
                logger.info(f"  Minimum sphere radius: {radius:.4f} m ({radius/wavelength:.1f} lambda)")
                return radius
    except Exception as e:
        logger.warning(f"Could not estimate radius: {e}")
    
    # Fallback
    radius = 15 * wavelength  # Conservative for unknown antenna
    logger.info(f"Using fallback radius: {radius:.4f} m ({radius/wavelength:.1f} lambda)")
    return radius