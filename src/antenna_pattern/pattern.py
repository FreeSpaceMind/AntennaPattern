"""
Core class for antenna pattern representation and manipulation.
"""
import numpy as np
import xarray as xr
from typing import Optional, Union, Tuple, Dict, Any, Set, Generator
import logging
from pathlib import Path
from contextlib import contextmanager

from .pattern_functions import(
    unwrap_phase, swap_polarization_axes, apply_mars, 
    normalize_phase, change_polarization, translate,
    scale_pattern, transform_coordinates, mirror_pattern,
    normalize_at_boresight, shift_theta_origin,
    shift_phi_origin
)
from .utilities import find_nearest
from .polarization import (
    polarization_tp2xy, polarization_tp2rl
)
from .analysis import (
    calculate_phase_center, get_axial_ratio,
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
        metadata (Dict[str, Any]): Optional metadata for the pattern including operations history
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
                 polarization: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an AntennaPattern with the given parameters.
        
        Args:
            theta: Array of theta angles in degrees
            phi: Array of phi angles in degrees
            frequency: Array of frequencies in Hz
            e_theta: Complex array of e_theta values [freq, theta, phi]
            e_phi: Complex array of e_phi values [freq, theta, phi]
            polarization: Optional polarization type. If None, determined automatically.
            metadata: Optional metadata dictionary
            
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
        
        # Initialize metadata if provided
        self.metadata = metadata.copy() if metadata is not None else {'operations': []}

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
            polarization=self.polarization,
            metadata={'parent_pattern': 'Single frequency view'}
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

        # Update metadata
        if self.metadata is not None:
            self.metadata['polarization'] = standard_pol
    
    def change_polarization(self, new_polarization: str) -> None:
        """
        Change the polarization of the antenna pattern.
        
        Args:
            new_polarization: New polarization type to use
            
        Raises:
            ValueError: If the new polarization is invalid
        """
        # Delegate to the pattern_functions implementation
        change_polarization(self, new_polarization)
    
    def translate(self, translation: np.ndarray, normalize: bool = True) -> None:
        """
        Shifts the antenna phase pattern to place the origin of the coordinate 
        system at the location defined by the translation.
        
        Args:
            translation: [x, y, z] translation vector in meters. Can be 1D (applied to
                all frequencies) or 2D with shape (num_frequencies, 3).
            normalize: if true, normalize the translated phase pattern to zero degrees
            
        Raises:
            ValueError: If translation has incorrect shape
        """
        # Delegate to the pattern_functions implementation
        translate(self, translation, normalize)
    
    def find_phase_center(self, theta_angle: float, frequency: Optional[float] = None, 
                        n_iter: int = 10) -> np.ndarray:
        """
        Finds the optimum phase center given a theta angle and frequency.
        
        The optimum phase center is the point that, when used as the origin,
        minimizes the phase variation across the beam from -theta_angle to +theta_angle.
        Uses basinhopping optimization to find the global minimum.
        
        Args:
            theta_angle: Angle in degrees to optimize phase center for
            frequency: Optional specific frequency to use, or None to use first frequency
            method: Optimization method ('spread', 'flatness', or 'combined')
                - 'spread': Minimize max-min phase difference at each theta angle across phi cuts
                - 'flatness': Minimize max-min phase difference across the entire theta-phi region
                - 'combined': Weighted combination of both spread and flatness metrics (default)
            outlier_threshold: Standard deviation threshold for outlier removal
            n_iter: Number of iterations for basinhopping
            spread_weight: Weight for spread component in combined metric (0.0-1.0)
                - 0.0: Pure flatness optimization
                - 1.0: Pure spread optimization
                - Default 0.5: Equal weighting of both metrics
            
        Returns:
            np.ndarray: [x, y, z] coordinates of the optimum phase center
        """
        # Delegate to the analysis.py implementation
        return calculate_phase_center(
            self, theta_angle, frequency, n_iter
        )

    def shift_to_phase_center(self, theta_angle: float, frequency: Optional[float] = None,
                            method: str = 'combined', outlier_threshold: float = 3.0,
                            n_iter: int = 10, spread_weight: float = 0.5) -> np.ndarray:
        """
        Find the phase center and shift the pattern to it.
        
        Args:
            theta_angle: Angle in degrees to optimize phase center for
            frequency: Optional specific frequency to use, or None to use first frequency
            method: Optimization method ('spread', 'flatness', or 'combined')
            outlier_threshold: Standard deviation threshold for outlier removal
            n_iter: Number of iterations for basinhopping
            spread_weight: Weight for spread component when method='combined' (0.0-1.0)
            
        Returns:
            np.ndarray: The translation vector used
        """  
        
        # Calculate phase center without modifying the pattern
        translation = calculate_phase_center(
            self, theta_angle, frequency, method, outlier_threshold, n_iter, spread_weight
        )
        
        # Then apply the translation
        self.translate(translation)
        
        # Update metadata if needed
        if hasattr(self, 'metadata') and self.metadata is not None:
            if 'operations' not in self.metadata:
                self.metadata['operations'] = []
            self.metadata['operations'].append({
                'type': 'shift_to_phase_center',
                'theta_angle': theta_angle,
                'frequency': frequency,
                'translation': translation.tolist() if hasattr(translation, 'tolist') else translation,
                'method': method,
                'outlier_threshold': outlier_threshold,
                'n_iter': n_iter,
                'spread_weight': spread_weight
            })
        
        return translation
    
    def normalize_phase(self, reference_theta=0, reference_phi=0) -> None:
        """
        Normalize the phase of the antenna pattern based on its polarization type.
        
        This function sets the phase of the co-polarized component at the reference
        point (closest to reference_theta, reference_phi) to zero, while preserving
        the relative phase between components.
        
        Args:
            reference_theta: Reference theta angle in degrees (default: 0)
            reference_phi: Reference phi angle in degrees (default: 0)
        """
        # Delegate to the pattern_functions implementation
        normalize_phase(self, reference_theta, reference_phi)

    def normalize_at_boresight(self) -> None:
        """
        Normalize the phase and magnitude of the antenna pattern so that both e_theta 
        and e_phi components for all phi cuts have the same phase and magnitude at 
        boresight (theta=0). Normalized to the first phi cut (typically phi =0).
        
        Raises:
            ValueError: If the pattern doesn't have a theta=0 point
        """
        # Delegate to the pattern_functions implementation
        normalize_at_boresight(self)

    def apply_mars(self, maximum_radial_extent: float) -> None:
        """
        Apply Mathematical Absorber Reflection Suppression technique.
        
        The MARS algorithm transforms antenna measurement data to mitigate reflections
        from the measurement chamber. It is particularly effective for electrically
        large antennas.
        
        Args:
            maximum_radial_extent: Maximum radial extent of the antenna in meters
        """
        # Delegate to the pattern_functions implementation
        apply_mars(self, maximum_radial_extent)
    
    def swap_polarization_axes(self) -> None:
        """
        Swap vertical and horizontal polarization ports.
        """
        # Delegate to the pattern_functions implementation
        swap_polarization_axes(self)
    
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
        
        return np.degrees(phase)
    
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
    
    def scale_pattern(self, scale_db: Union[float, np.ndarray], 
                      freq_scale: Optional[np.ndarray] = None,
                      phi_scale: Optional[np.ndarray] = None) -> None:
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
                
        Raises:
            ValueError: If input arrays have incompatible dimensions
        """
        # Delegate to the pattern_functions implementation
        scale_pattern(self, scale_db, freq_scale, phi_scale)
    
    def transform_coordinates(self, format: str = 'sided') -> None:
        """
        Transform pattern coordinates to conform to a specified theta/phi convention.
        
        This function rearranges the existing pattern data to match one of two standard
        coordinate conventions without interpolation:
        
        - 'sided': theta 0:180, phi 0:360 (spherical convention)
        - 'central': theta -180:180, phi 0:180 (more common for antenna patterns)
        
        Args:
            format: Target coordinate format ('sided' or 'central')
            
        Raises:
            ValueError: If format is not 'sided' or 'central'
        """
        # Delegate to the pattern_functions implementation
        transform_coordinates(self, format)

    def shift_theta_origin(self, theta_offset: float) -> None:
        """
        Shifts the origin of the theta coordinate axis for all phi cuts.
        
        This is useful for aligning measurement data when the mechanical 
        antenna rotation axis doesn't align with the desired coordinate 
        system (e.g., antenna boresight).
        
        Args:
            theta_offset: Angle in degrees to shift the theta origin.
                        Positive values move theta=0 to the right (positive theta),
                        negative values move theta=0 to the left (negative theta).
                        
        Notes:
            - This performs interpolation along the theta axis for each phi cut
            - Original theta grid points are preserved
        """
        # Delegate to the pattern_functions implementation
        shift_theta_origin(self, theta_offset)

    def shift_phi_origin(self, phi_offset: float) -> None:
        """
        Shifts the origin of the phi coordinate axis for the pattern.
        
        This is useful for aligning measurement data when the mechanical 
        antenna rotation reference doesn't align with the desired coordinate
        system (e.g., principal planes).
        
        Args:
            phi_offset: Angle in degrees to shift the phi origin.
                    Positive values rotate phi clockwise,
                    negative values rotate phi counterclockwise.
                    
        Notes:
            - This performs interpolation along the phi axis for each theta value
            - Original phi grid points are preserved
            - Takes into account the periodicity of phi (0° = 360°)
        """
        # Delegate to the pattern_functions implementation
        shift_phi_origin(self, phi_offset)

    def split_patterns(self) -> Tuple['AntennaPattern', 'AntennaPattern']:
        """
        Split antenna pattern into two separate patterns if both conditions are met:
        1. Pattern has both negative and positive theta values
        2. Phi range is greater than 180 degrees
        
        The function splits the pattern along the phi axis, creating two patterns
        with phi ranges of 0:180 degrees. The theta values remain unchanged.
        
        Returns:
            Tuple[AntennaPattern, AntennaPattern]: Tuple containing two separated patterns
            
        Raises:
            ValueError: If pattern does not meet the conditions for splitting
        """
        # Check condition 1: Pattern has both negative and positive theta values
        theta = self.theta_angles
        has_negative_theta = np.any(theta < 0)
        has_positive_theta = np.any(theta > 0)
        
        if not (has_negative_theta and has_positive_theta):
            raise ValueError("Pattern does not have both negative and positive theta values")
        
        # Check condition 2: Phi range is greater than 180 degrees
        phi = self.phi_angles
        phi_range = np.max(phi) - np.min(phi)
        
        if phi_range <= 180:
            raise ValueError("Pattern phi range is not greater than 180 degrees")
        
        # Extract data
        e_theta = self.data.e_theta.values
        e_phi = self.data.e_phi.values
        frequency = self.frequencies
        
        # Determine how to split the phi range
        # First find which indices are in range 0-180
        first_range_mask = (phi >= 0) & (phi < 180)
        
        # If no indices fall in this range, we need to shift the phi values
        if not np.any(first_range_mask):
            # Shift phi values to be in range 0-360
            phi_shifted = np.mod(phi, 360)
            first_range_mask = (phi_shifted >= 0) & (phi_shifted <= 180)
        
        # Second range includes everything outside first range, but shifted to 0-180
        second_range_mask = ~first_range_mask
        
        # Get indices for each range
        first_indices = np.where(first_range_mask)[0]
        second_indices = np.where(second_range_mask)[0]
        
        if len(first_indices) == 0 or len(second_indices) == 0:
            raise ValueError("Could not properly separate phi values into two ranges")
        
        # Create phi arrays for the two new patterns
        phi1 = phi[first_indices]
        
        # For phi2, we need to normalize values to 0-180 range
        phi2_original = phi[second_indices]
        phi2 = np.mod(phi2_original, 360)
        phi2 = np.where(phi2 >= 180, phi2 - 360, phi2)
        phi2 = np.where(phi2 <= 0, phi2 + 180, phi2)
        phi2 = np.sort(phi2)  # Sort to ensure ascending order
        
        # Create field component arrays for the two patterns
        e_theta1 = e_theta[:, :, first_indices]
        e_phi1 = e_phi[:, :, first_indices]
        
        e_theta2 = -np.flip(e_theta[:, :, second_indices], axis=1)
        e_phi2 = -np.flip(e_phi[:, :, second_indices], axis=1)
        
        # Create new AntennaPattern objects
        pattern1 = AntennaPattern(
            theta=theta,
            phi=phi1,
            frequency=frequency,
            e_theta=e_theta1,
            e_phi=e_phi1,
            polarization=self.polarization,
            metadata={
                'parent_pattern': 'Split pattern 1',
                'original_phi_indices': first_indices.tolist()
            } if self.metadata else None
        )
        
        pattern2 = AntennaPattern(
            theta=theta,
            phi=phi2,
            frequency=frequency,
            e_theta=e_theta2,
            e_phi=e_phi2,
            polarization=self.polarization,
            metadata={
                'parent_pattern': 'Split pattern 2',
                'original_phi_indices': second_indices.tolist()
            } if self.metadata else None
        )
        
        return pattern1, pattern2
    
    def mirror(self) -> None:
        """
        Mirror the antenna pattern about theta=0.
        
        This function reflects the pattern data across the theta=0 plane,
        effectively mirroring the pattern. It's useful for creating symmetric patterns
        or for fixing incomplete measurement data.
        
        Notes:
            If the pattern does not include theta=0, the function will raise a ValueError.
            The pattern should have theta values in [-180, 180] range.
        """
        # Delegate to the pattern_functions implementation
        mirror_pattern(self)