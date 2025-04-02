"""
Core class for antenna pattern representation and manipulation.
"""
import numpy as np
import xarray as xr
from typing import Optional, Union, Tuple, Dict, Any, Set, Generator
import logging
from pathlib import Path
from contextlib import contextmanager

from .utilities import find_nearest, unwrap_phase, scale_amplitude
from .polarization import (
    polarization_tp2xy, polarization_xy2pt, polarization_tp2rl, 
    polarization_rl2xy, polarization_rl2tp
)
from .analysis import translate_phase_pattern, find_phase_center, apply_mars, get_axial_ratio, calculate_beamwidth

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
            from scipy.interpolate import interp1d
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
            
            # Use more robust griddata interpolation
            from scipy.interpolate import griddata
            
            # Create meshgrid of input frequencies and phi angles
            freq_grid_in, phi_grid_in = np.meshgrid(freq_scale, phi_scale, indexing='ij')
            points_in = np.column_stack((freq_grid_in.ravel(), phi_grid_in.ravel()))
            values_in = scale_db.ravel()
            
            # Set reasonable limits for dB values to prevent overflow
            max_db_value = 50.0
            min_db_value = -50.0
            
            # Create a 3D array to store scaling factors for each point in the pattern
            # This ensures we apply the exact same scaling to both e_theta and e_phi
            # at each point, preserving polarization characteristics
            scale_factors = np.zeros((n_freq, n_theta, n_phi))
            
            # For each phi cut, interpolate the scaling value at each frequency
            for p_idx, phi_val in enumerate(pattern_phi):
                # Create interpolation points for this phi value
                points_out = np.column_stack((pattern_freq, np.full_like(pattern_freq, phi_val)))
                
                # Interpolate scaling values for this phi angle at each frequency
                scale_vals = griddata(points_in, values_in, points_out, 
                                    method='linear', fill_value=0.0)
                
                # Clip to reasonable range
                scale_vals = np.clip(scale_vals, min_db_value, max_db_value)
                
                # Replace any NaN values
                scale_vals = np.nan_to_num(scale_vals, nan=0.0)
                
                # Assign to all theta values for this phi angle
                for f_idx in range(n_freq):
                    scale_factors[f_idx, :, p_idx] = scale_vals[f_idx]
            
            # Log warning if extreme values were detected
            if np.any(scale_factors >= max_db_value) or np.any(scale_factors <= min_db_value):
                logger.warning("Extreme scaling values detected in interpolation and clipped to range [%f, %f] dB", 
                            min_db_value, max_db_value)
            
            # Convert dB to linear scale factors
            linear_scale_factors = 10**(scale_factors / 20.0)
            
            # Apply scaling using numpy broadcasting - this is more efficient
            # and ensures the same scaling is applied to both field components
            e_theta_scaled = self.data.e_theta.values * linear_scale_factors
            e_phi_scaled = self.data.e_phi.values * linear_scale_factors
            
            return AntennaPattern(
                theta=self.theta_angles,
                phi=pattern_phi,
                frequency=pattern_freq,
                e_theta=e_theta_scaled,
                e_phi=e_phi_scaled,
                polarization=self.polarization
            )
        
        raise ValueError("Invalid scale_db format. Must be scalar, 1D or 2D array.")
    
    def write_cut(self, file_path: Union[str, Path], polarization_format: int = 1) -> None:
        """
        Write the antenna pattern to a CUT file.
        
        Args:
            file_path: Path to the output CUT file
            polarization_format: Format for polarization components
                1 = Linear theta and phi
                2 = Right and left hand circular
                3 = Linear co and cross (Ludwig's 3rd definition)
                
        Raises:
            ValueError: If polarization_format is invalid
            IOError: If file cannot be written
        """
        # Extract data
        theta = self.data.theta.values
        phi = self.data.phi.values
        frequency = self.data.frequency.values
        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        
        # Set polarization components based on format
        if polarization_format == 1:
            e_co = e_theta
            e_cx = e_phi
        elif polarization_format == 2:
            e_co, e_cx = polarization_tp2rl(phi, e_theta, e_phi)
        elif polarization_format == 3:
            e_co, e_cx = polarization_tp2xy(phi, e_theta, e_phi)
        else:
            raise ValueError(f"Invalid polarization format: {polarization_format}. Must be 1, 2, or 3.")

        # Calculate frequency increment
        freq_num = len(frequency)
        theta_length = len(theta)
        phi_length = len(phi)

        # Ensure file_path is a Path object
        file_path = Path(file_path)
        
        with open(file_path, "w") as writer:
            for f_idx in range(freq_num):
                for p_idx in range(phi_length):
                    writer.write(f"{frequency[f_idx]/1e6}MHz\n")
                    # Write header for each phi cut
                    theta_start = theta[0]
                    theta_increment = theta[1] - theta[0] if len(theta) > 1 else 0.0
                    header = f"{theta_start} {theta_increment} {theta_length} {phi[p_idx]} {polarization_format} 1 2\n"
                    writer.write(header)

                    # Write data lines for each theta value
                    for t_idx in range(theta_length):
                        YA = e_co[f_idx, t_idx, p_idx]
                        YB = e_cx[f_idx, t_idx, p_idx]
                        line = f"{YA.real} {YA.imag} {YB.real} {YB.imag}\n"
                        writer.write(line)