"""
Core class for antenna pattern representation and manipulation.
"""
import numpy as np
import xarray as xr
from typing import Optional, Union, Tuple, Dict, Any, Set, Generator
import logging
from pathlib import Path
from contextlib import contextmanager
from scipy.interpolate import RegularGridInterpolator, interp1d

from .utilities import (
        find_nearest, unwrap_phase, scale_amplitude,
        transform_tp2uvw, isometric_rotation, transform_uvw2tp
        )
from .polarization import (
    polarization_tp2xy, polarization_xy2pt, polarization_tp2rl, 
    polarization_rl2xy, polarization_rl2tp
)
from .analysis import (
    translate_phase_pattern, find_phase_center, apply_mars, get_axial_ratio, 
    calculate_beamwidth, normalize_phase
    )

# Configure logging
logger = logging.getLogger(__name__)

class AntennaPattern:
    """
    A class to represent antenna far field patterns.
    
    This class encapsulates antenna pattern data and provides methods for manipulation, 
    analysis and conversion between different formats and coordinate systems.
    
    Attributes:
        data (xarray.Dataset): The core dataset containing all pattern information
            with dimensions (frequency, theta, phi) and data variables:
            - e_theta: Complex theta polarization component
            - e_phi: Complex phi polarization component
            - e_co: Co-polarized component (determined by polarization attribute)
            - e_cx: Cross-polarized component (determined by polarization attribute)
        polarization (str): The polarization type ('rhcp', 'lhcp', 'x', 'y', 'theta', 'phi')
    """
    
    VALID_POLARIZATIONS: Set[str] = {
        'rhcp', 'rh', 'r',            # Right-hand circular
        'lhcp', 'lh', 'l',            # Left-hand circular
        'x', 'l3x',                   # Linear X (Ludwig's 3rd)
        'y', 'l3y',                   # Linear Y (Ludwig's 3rd)
        'theta',                      # Spherical theta
        'phi'                         # Spherical phi
    }
    
    def __init__(self, 
                 theta: np.ndarray, 
                 phi: np.ndarray, 
                 frequency: np.ndarray,
                 e_theta: np.ndarray, 
                 e_phi: np.ndarray,
                 polarization: Optional[str] = None):
        """
        Initialize an AntennaPattern with the given parameters.
        
        Args:
            theta: Array of theta angles in degrees
            phi: Array of phi angles in degrees
            frequency: Array of frequencies in Hz
            e_theta: Complex array of e_theta values [freq, theta, phi]
            e_phi: Complex array of e_phi values [freq, theta, phi]
            polarization: Optional polarization type. If None, determined automatically.
            
        Raises:
            ValueError: If arrays have incompatible dimensions
            ValueError: If polarization is invalid
        """
        # Convert arrays to efficient dtypes for faster processing
        e_theta = np.asarray(e_theta, dtype=np.complex64)
        e_phi = np.asarray(e_phi, dtype=np.complex64)

        # Validate array dimensions
        expected_shape = (len(frequency), len(theta), len(phi))
        if e_theta.shape != expected_shape:
            raise ValueError(f"e_theta shape mismatch: expected {expected_shape}, got {e_theta.shape}")
        if e_phi.shape != expected_shape:
            raise ValueError(f"e_phi shape mismatch: expected {expected_shape}, got {e_phi.shape}")
        
        # Create core dataset
        self.data = xr.Dataset(
            data_vars={
                'e_theta': (('frequency', 'theta', 'phi'), e_theta),
                'e_phi': (('frequency', 'theta', 'phi'), e_phi),
            },
            coords={
                'theta': theta,
                'phi': phi,
                'frequency': frequency,
            }
        )
        
        # Assign polarization and compute derived components
        self.assign_polarization(polarization)
        
        # Initialize cache
        self._cache: Dict[str, Any] = {}
    
    @property
    def frequencies(self) -> np.ndarray:
        """Get frequencies in Hz."""
        return self.data.frequency.values
    
    @property
    def theta_angles(self) -> np.ndarray:
        """Get theta angles in degrees."""
        return self.data.theta.values
    
    @property
    def phi_angles(self) -> np.ndarray:
        """Get phi angles in degrees."""
        return self.data.phi.values
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache = {}
    
    @contextmanager
    def at_frequency(self, frequency: float) -> 'Generator[AntennaPattern, None, None]':
        """
        Context manager to temporarily extract a single-frequency pattern.
        
        Args:
            frequency: Frequency in Hz
            
        Yields:
            AntennaPattern: Single-frequency pattern
            
        Example:
            ```python
            with pattern.at_frequency(2.4e9) as single_freq_pattern:
                # Work with single_freq_pattern
            ```
        """
        freq_value, freq_idx = find_nearest(self.frequencies, frequency)
        
        # Extract data for the nearest frequency
        single_freq_data = self.data.isel(frequency=freq_idx)
        
        # Create a new pattern with the extracted data
        single_freq_pattern = AntennaPattern(
            theta=self.theta_angles,
            phi=self.phi_angles,
            frequency=np.array([freq_value]),
            e_theta=np.expand_dims(single_freq_data.e_theta.values, axis=0),
            e_phi=np.expand_dims(single_freq_data.e_phi.values, axis=0),
            polarization=self.polarization
        )
        
        yield single_freq_pattern
    
    def assign_polarization(self, polarization: Optional[str] = None) -> None:
        """
        Assign a polarization to the antenna pattern and compute e_co and e_cx.
        
        If polarization is None, it is automatically determined based on which 
        polarization component has the highest peak gain.
        
        Args:
            polarization: Polarization type or None to auto-detect
            
        Raises:
            ValueError: If the specified polarization is invalid
        """
        # Get underlying numpy arrays for calculations
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        
        # Calculate different polarization components
        e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
        e_r, e_l = polarization_tp2rl(phi, e_theta, e_phi)
        
        # Auto-detect polarization if not specified
        if polarization is None:
            e_x_max = np.max(np.abs(e_x))
            e_y_max = np.max(np.abs(e_y))
            e_r_max = np.max(np.abs(e_r))
            e_l_max = np.max(np.abs(e_l))
            
            max_val = max(e_x_max, e_y_max, e_r_max, e_l_max)
            
            if e_x_max == max_val:
                polarization = "x"
            elif e_y_max == max_val:
                polarization = "y"
            elif e_r_max == max_val:
                polarization = "rhcp"
            elif e_l_max == max_val:
                polarization = "lhcp"
        
        # Map variations to standard polarization names
        pol_lower = polarization.lower() if polarization else ""
        
        if pol_lower in {"rhcp", "rh", "r"}:
            e_co, e_cx = e_r, e_l
            standard_pol = "rhcp"
        elif pol_lower in {"lhcp", "lh", "l"}:
            e_co, e_cx = e_l, e_r
            standard_pol = "lhcp"
        elif pol_lower in {"x", "l3x"}:
            e_co, e_cx = e_x, e_y
            standard_pol = "x"
        elif pol_lower in {"y", "l3y"}:
            e_co, e_cx = e_y, e_x
            standard_pol = "y"
        elif pol_lower == "theta":
            e_co, e_cx = e_theta, e_phi
            standard_pol = "theta"
        elif pol_lower == "phi":
            e_co, e_cx = e_phi, e_theta
            standard_pol = "phi"
        else:
            raise ValueError(f"Invalid polarization: {polarization}")
        
        # Store polarization components and type
        self.data['e_co'] = (('frequency', 'theta', 'phi'), e_co)
        self.data['e_cx'] = (('frequency', 'theta', 'phi'), e_cx)
        self.polarization = standard_pol
    
    def change_polarization(self, new_polarization: str) -> 'AntennaPattern':
        """
        Create a new pattern with a different polarization assignment.
        
        Args:
            new_polarization: New polarization type to use
        
        Returns:
            AntennaPattern: New pattern with the specified polarization
            
        Raises:
            ValueError: If the new polarization is invalid
        """

        # Create a new pattern with the same data but different polarization
        return AntennaPattern(
            theta=self.theta_angles,
            phi=self.phi_angles,
            frequency=self.frequencies,
            e_theta=self.data.e_theta.values,
            e_phi=self.data.e_phi.values,
            polarization=new_polarization
        )
    
    def translate(self, translation: np.ndarray) -> 'AntennaPattern':
        """
        Shifts the antenna phase pattern to place the origin of the coordinate 
        system at the location defined by the translation.
        
        Args:
            translation: [x, y, z] translation vector in meters. Can be 1D (applied to
                all frequencies) or 2D with shape (num_frequencies, 3).
        
        Returns:
            AntennaPattern: A new AntennaPattern with translated phase
            
        Raises:
            ValueError: If translation has incorrect shape
        """
        # Delegate to the analysis module
        return translate_phase_pattern(self, translation)
    
    def find_phase_center(self, theta_angle: float, frequency: Optional[float] = None) -> np.ndarray:
        """
        Finds the optimum phase center given a theta angle and frequency.
        
        The optimum phase center is the point that, when used as the origin,
        minimizes the phase variation across the beam from -theta_angle to +theta_angle.
        
        Args:
            theta_angle: Angle in degrees to optimize phase center for
            frequency: Optional specific frequency to use, or None to use all frequencies
        
        Returns:
            np.ndarray: [x, y, z] coordinates of the optimum phase center
        """
        return find_phase_center(self, theta_angle, frequency)
    
    def shift_to_phase_center(self, theta_angle: float, frequency: Optional[float] = None) -> Tuple['AntennaPattern', np.ndarray]:
        """
        Find the phase center and shift the pattern to it.
        
        Args:
            theta_angle: Angle in degrees to optimize phase center for
            frequency: Optional specific frequency to use, or None to use all frequencies
            
        Returns:
            Tuple[AntennaPattern, np.ndarray]: Shifted pattern and the translation vector
        """
        translation = self.find_phase_center(theta_angle, frequency)
        return self.translate(translation), translation
    
    def normalize_phase(self, reference_theta=0, reference_phi=0):
        """
        Normalize the phase of an antenna pattern based on its polarization type.
        
        This function sets the phase of the co-polarized component at the reference
        point (closest to reference_theta, reference_phi) to zero, while preserving
        the relative phase between components.
        
        Args:
            reference_theta: Reference theta angle in degrees (default: 0)
            reference_phi: Reference phi angle in degrees (default: 0)
            
        Returns:
            AntennaPattern: New pattern with normalized phase
        """
        return normalize_phase(self, reference_theta, reference_phi)

    def apply_mars(self, maximum_radial_extent: float) -> 'AntennaPattern':
        """
        Apply Mathematical Absorber Reflection Suppression technique.
        
        The MARS algorithm transforms antenna measurement data to mitigate reflections
        from the measurement chamber. It is particularly effective for electrically
        large antennas.
        
        Args:
            maximum_radial_extent: Maximum radial extent of the antenna in meters
            
        Returns:
            AntennaPattern: New pattern with MARS algorithm applied
        """

        return apply_mars(self, maximum_radial_extent)
    
    def swap_polarization_axes(self) -> 'AntennaPattern':
        """
        Swap vertical and horizontal polarization ports.
        
        Returns:
            AntennaPattern: New pattern with swapped polarization
        """
        phi = self.data.phi.values
        e_theta = self.data.e_theta.values 
        e_phi = self.data.e_phi.values
        
        # Convert to x/y and back to swap the axes
        e_x, e_y = polarization_tp2xy(phi, e_theta, e_phi)
        e_theta_new, e_phi_new = polarization_xy2pt(phi, e_y, e_x)  # Note: x and y are swapped
        
        return AntennaPattern(
            theta=self.theta_angles,
            phi=phi,
            frequency=self.frequencies,
            e_theta=e_theta_new,
            e_phi=e_phi_new,
            polarization=self.polarization
        )
    
    def get_gain_db(self, component: str = 'e_co') -> xr.DataArray:
        """
        Get gain in dB for a specific field component.
        
        Args:
            component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')
            
        Returns:
            xr.DataArray: Gain in dB
            
        Raises:
            KeyError: If component does not exist
        """
        cache_key = f"gain_db_{component}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        result = 20 * np.log10(np.abs(self.data[component]))
        self._cache[cache_key] = result
        return result
    
    def get_phase(self, component: str = 'e_co', unwrapped: bool = False) -> xr.DataArray:
        """
        Get phase for a specific field component.
        
        Args:
            component: Field component ('e_co', 'e_cx', 'e_theta', 'e_phi')
            unwrapped: If True, return unwrapped phase (no 2π discontinuities)
            
        Returns:
            xr.DataArray: Phase in radians
            
        Raises:
            KeyError: If component does not exist
        """
        if component not in self.data:
            raise KeyError(f"Component {component} not found in pattern data. "
                          f"Available: {list(self.data.data_vars.keys())}")
        
        phase = np.angle(self.data[component])
        
        if unwrapped:
            # Unwrap along theta dimension to remove discontinuities
            phase = xr.apply_ufunc(
                unwrap_phase,
                phase,
                input_core_dims=[["theta"]],
                output_core_dims=[["theta"]],
                vectorize=True
            )
        
        return phase
    
    def get_polarization_ratio(self) -> xr.DataArray:
        """
        Calculate the ratio of cross-pol to co-pol in dB.
        
        Returns:
            xr.DataArray: Cross-pol/co-pol ratio in dB
        """
        co_mag = np.abs(self.data.e_co)
        cx_mag = np.abs(self.data.e_cx)
        
        # Prevent division by zero
        co_mag = xr.where(co_mag < 1e-15, 1e-15, co_mag)
        
        return 20 * np.log10(cx_mag / co_mag)
    
    def get_axial_ratio(self) -> xr.DataArray:
        """
        Calculate the axial ratio (ratio of major to minor axis of polarization ellipse).
        
        Returns:
            xr.DataArray: Axial ratio (linear scale)
        """

        return get_axial_ratio(self)
    
    def calculate_beamwidth(self, frequency: Optional[float] = None, level_db: float = -3.0) -> Dict[str, float]:
        """
        Calculate the beamwidth at specified level for principal planes.
        
        Args:
            frequency: Optional frequency to calculate beamwidth for, or None for all
            level_db: Level relative to maximum at which to measure beamwidth (default: -3 dB)
            
        Returns:
            Dict[str, float]: Beamwidths in degrees for E and H planes
        """

        return calculate_beamwidth(self, frequency, level_db)
    
    def scale_pattern(self, scale_db: Union[float, np.ndarray], 
                    freq_scale: Optional[np.ndarray] = None,
                    phi_scale: Optional[np.ndarray] = None) -> 'AntennaPattern':
        """
        Scale the amplitude of the antenna pattern by the input value in dB.
        
        This method supports several input formats:
        1. A scalar value: Apply the same scaling to all frequencies and angles
        2. A 1D array matching frequency length: Apply frequency-dependent scaling
        3. A 1D array with custom frequencies: Interpolate to pattern frequencies
        4. A 2D array with scaling values for freq/phi combinations
        
        Args:
            scale_db: Scaling values in dB. Can be:
                - float: Single value applied to all frequencies and angles
                - 1D array[freq]: Values for each frequency in pattern
                - 2D array[freq, phi]: Values for each frequency/phi combination
            freq_scale: Optional frequency vector in Hz when scale_db doesn't match
                pattern frequency points. Required if scale_db is 1D and doesn't 
                match pattern frequencies, or if scale_db is 2D.
            phi_scale: Optional phi angle vector in degrees when scale_db is 2D and 
                doesn't match pattern phi points.
                
        Returns:
            AntennaPattern: New pattern with scaled amplitude
            
        Raises:
            ValueError: If input arrays have incompatible dimensions
        """
        # Get pattern dimensions
        pattern_freq = self.frequencies
        pattern_phi = self.phi_angles
        pattern_theta = self.theta_angles
        n_freq = len(pattern_freq)
        n_phi = len(pattern_phi)
        n_theta = len(pattern_theta)
        
        # Case 1: Single scalar value - apply uniformly
        if np.isscalar(scale_db):
            # Create a new pattern with scaled field components
            return AntennaPattern(
                theta=pattern_theta,
                phi=pattern_phi,
                frequency=pattern_freq,
                e_theta=scale_amplitude(self.data.e_theta.values, scale_db),
                e_phi=scale_amplitude(self.data.e_phi.values, scale_db),
                polarization=self.polarization
            )
        
        # Convert to numpy array if not already
        scale_db = np.asarray(scale_db)
        
        # Case 2: 1D array matching frequency length
        if scale_db.ndim == 1 and len(scale_db) == n_freq and freq_scale is None:
            # Reshape for broadcasting - add axes for theta and phi dimensions
            scale_factor = scale_db.reshape(-1, 1, 1)
            
            return AntennaPattern(
                theta=pattern_theta,
                phi=pattern_phi,
                frequency=pattern_freq,
                e_theta=scale_amplitude(self.data.e_theta.values, scale_factor),
                e_phi=scale_amplitude(self.data.e_phi.values, scale_factor),
                polarization=self.polarization
            )
        
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
            
            return AntennaPattern(
                theta=pattern_theta,
                phi=pattern_phi,
                frequency=pattern_freq,
                e_theta=scale_amplitude(self.data.e_theta.values, scale_factor),
                e_phi=scale_amplitude(self.data.e_phi.values, scale_factor),
                polarization=self.polarization
            )
        
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
            
            # Log warning if extreme values were detected
            if np.any(scale_factors >= max_db_value) or np.any(scale_factors <= min_db_value):
                logger.warning("Extreme scaling values detected and clipped to range [%f, %f] dB", 
                            min_db_value, max_db_value)
            
            # Convert dB to linear scale factors
            linear_scale_factors = 10**(scale_factors / 20.0)
            
            # Apply scaling using numpy broadcasting
            e_theta_scaled = self.data.e_theta.values * linear_scale_factors
            e_phi_scaled = self.data.e_phi.values * linear_scale_factors
            
            return AntennaPattern(
                theta=pattern_theta,
                phi=pattern_phi,
                frequency=pattern_freq,
                e_theta=e_theta_scaled,
                e_phi=e_phi_scaled,
                polarization=self.polarization
            )
        
        raise ValueError("Invalid scale_db format. Must be scalar, 1D or 2D array.")
    
    def rotate(self, azimuth: float, elevation: float, roll: float) -> 'AntennaPattern':
        """
        Rotate the antenna pattern by specified angles.
        
        Applies an isometric rotation to the antenna pattern using rotation angles
        in the given order: roll (around z-axis), then elevation (around x-axis), 
        then azimuth (around y-axis).
        
        Coordinate system:
        - z-axis is boresight (theta=0)
        - x-axis corresponds to theta=90, phi=0
        - y-axis corresponds to theta=90, phi=90
        
        Args:
            azimuth: Rotation around y-axis in degrees
            elevation: Rotation around x-axis in degrees
            roll: Rotation around z-axis in degrees
            
        Returns:
            AntennaPattern: New pattern with rotated coordinates
        """
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator
        import logging
        
        # Get pattern dimensions
        freq_array = self.frequencies
        theta_array = self.theta_angles
        phi_array = self.phi_angles
        
        # Create output arrays
        e_theta_rotated = np.zeros_like(self.data.e_theta.values)
        e_phi_rotated = np.zeros_like(self.data.e_phi.values)
        
        # Create meshgrid for original theta and phi angles
        THETA, PHI = np.meshgrid(theta_array, phi_array, indexing='ij')
        
        # Convert grid to direction cosines
        u_out, v_out, w_out = transform_tp2uvw(THETA, PHI)
        
        # Apply inverse rotation to find where each point came from
        u_in, v_in, w_in = isometric_rotation(u_out, v_out, w_out, -azimuth, -elevation, -roll)
        
        # Convert back to theta-phi coordinates
        theta_in, phi_in = transform_uvw2tp(u_in, v_in, w_in)
        
        # Process each frequency
        for freq_idx, freq in enumerate(freq_array):
            # Extract original pattern data for this frequency
            e_theta_orig = self.data.e_theta.values[freq_idx]
            e_phi_orig = self.data.e_phi.values[freq_idx]
            
            # Normalize phi angles to the range expected by the interpolator
            phi_min, phi_max = np.min(phi_array), np.max(phi_array)
            phi_range = phi_max - phi_min
            
            # Handle edge case where phi wraps around
            if np.any(phi_in < phi_min) or np.any(phi_in > phi_max):
                phi_normalized = np.mod(phi_in - phi_min, phi_range) + phi_min
            else:
                phi_normalized = phi_in
            
            # Check which points have negative theta in input
            # This is specific to the issue with negative theta values
            negative_theta_input = theta_in < 0
            
            # To interpolate properly, we need to use absolute theta values
            # since the source data might only have positive theta
            theta_interp = np.abs(theta_in)
            
            # For points with negative theta, adjust phi
            # This is part of the fix for the negative theta phase issue
            if np.any(negative_theta_input):
                if np.isscalar(phi_normalized):
                    if negative_theta_input:
                        phi_normalized = (phi_normalized + 180) % 360
                else:
                    phi_normalized[negative_theta_input] = (phi_normalized[negative_theta_input] + 180) % 360
            
            # Prepare points for interpolation using absolute theta
            points = np.column_stack((theta_interp.flatten(), phi_normalized.flatten()))
            
            # Create interpolators
            interp_theta_real = RegularGridInterpolator(
                (theta_array, phi_array),
                np.real(e_theta_orig),
                bounds_error=False,
                fill_value=0,
                method='linear'
            )
            
            interp_theta_imag = RegularGridInterpolator(
                (theta_array, phi_array),
                np.imag(e_theta_orig),
                bounds_error=False,
                fill_value=0,
                method='linear'
            )
            
            interp_phi_real = RegularGridInterpolator(
                (theta_array, phi_array),
                np.real(e_phi_orig),
                bounds_error=False,
                fill_value=0,
                method='linear'
            )
            
            interp_phi_imag = RegularGridInterpolator(
                (theta_array, phi_array),
                np.imag(e_phi_orig),
                bounds_error=False,
                fill_value=0,
                method='linear'
            )
            
            # Apply interpolation
            theta_real = interp_theta_real(points).reshape(theta_in.shape)
            theta_imag = interp_theta_imag(points).reshape(theta_in.shape)
            phi_real = interp_phi_real(points).reshape(theta_in.shape)
            phi_imag = interp_phi_imag(points).reshape(theta_in.shape)
            
            # For negative theta points, we need to apply specific phase corrections
            # to the field components
            if np.any(negative_theta_input):
                # For points with negative theta, we need to negate both field components
                # This is the critical fix for the 180° phase shift issue
                e_theta_complex = theta_real + 1j * theta_imag
                e_phi_complex = phi_real + 1j * phi_imag
                
                # Apply 180° phase shift and reverse field component directions
                # for negative theta points
                e_theta_complex[negative_theta_input] = -e_theta_complex[negative_theta_input]
                e_phi_complex[negative_theta_input] = -e_phi_complex[negative_theta_input]
                
                # Update real and imaginary components
                theta_real = np.real(e_theta_complex)
                theta_imag = np.imag(e_theta_complex)
                phi_real = np.real(e_phi_complex)
                phi_imag = np.imag(e_phi_complex)
            
            # Special handling for points at or very near theta=0 and theta=180 (poles)
            near_poles = (np.abs(theta_in) < 1e-5) | (np.abs(theta_in - 180) < 1e-5)
            
            if np.any(near_poles):
                # Apply field component rotation at poles
                for i in range(THETA.shape[0]):
                    for j in range(THETA.shape[1]):
                        if near_poles[i, j]:
                            # For poles, the field components need to rotate based on phi angle
                            phi_diff = phi_normalized[i, j] - phi_array[j]
                            phi_diff_rad = np.radians(phi_diff)
                            
                            # Apply field component rotation
                            cos_phi = np.cos(phi_diff_rad)
                            sin_phi = np.sin(phi_diff_rad)
                            
                            e_th = complex(theta_real[i, j], theta_imag[i, j])
                            e_ph = complex(phi_real[i, j], phi_imag[i, j])
                            
                            # Rotate field components
                            e_th_new = e_th * cos_phi - e_ph * sin_phi
                            e_ph_new = e_th * sin_phi + e_ph * cos_phi
                            
                            # Store rotated components
                            theta_real[i, j] = np.real(e_th_new)
                            theta_imag[i, j] = np.imag(e_th_new)
                            phi_real[i, j] = np.real(e_ph_new)
                            phi_imag[i, j] = np.imag(e_ph_new)
            
            # Combine real and imaginary parts
            e_theta_rotated[freq_idx] = theta_real + 1j * theta_imag
            e_phi_rotated[freq_idx] = phi_real + 1j * phi_imag
    
        # Create new pattern with the rotated field components
        return AntennaPattern(
            theta=theta_array,
            phi=phi_array,
            frequency=freq_array,
            e_theta=e_theta_rotated,
            e_phi=e_phi_rotated,
            polarization=self.polarization
        )