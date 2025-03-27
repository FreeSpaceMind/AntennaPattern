"""
Core class for antenna pattern representation and manipulation.
"""
import numpy as np
import xarray as xr
from typing import Optional, Union, Tuple, Dict, Any, Set, Generator
import logging
from pathlib import Path
from contextlib import contextmanager

from .utilities import find_nearest, unwrap_phase
from .polarization import (
    polarization_tp2xy, polarization_xy2pt, polarization_tp2rl, 
    polarization_rl2xy, polarization_rl2tp
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
        from .analysis import translate_phase_pattern
        
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
        # Import here to avoid circular import
        from .analysis import find_phase_center
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
        # Import here to avoid circular import
        from .analysis import apply_mars
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
        # Import here to avoid circular import
        from .analysis import get_axial_ratio
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
        # Import here to avoid circular import
        from .analysis import calculate_beamwidth
        return calculate_beamwidth(self, frequency, level_db)
    
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