# AntennaPattern Package Reference

## Overview
AntennaPattern is a standalone package extracted from AntPy containing all antenna pattern analysis functionality. It handles reading/writing patterns, polarization conversions, phase manipulations, and various analysis methods.

## Package Structure
```
antenna_pattern/
├── __init__.py
├── pattern.py       # AntennaPattern class
├── io.py            # File I/O functions
├── polarization.py  # Polarization conversions
├── analysis.py      # Analysis functions
└── utilities.py     # Common utilities
```

## AntennaPattern Class

### Constructor
```python
AntennaPattern(
    theta: np.ndarray,               # Array of theta angles in degrees
    phi: np.ndarray,                 # Array of phi angles in degrees
    frequency: np.ndarray,           # Array of frequencies in Hz
    e_theta: np.ndarray,             # Complex array of e_theta values [freq, theta, phi]
    e_phi: np.ndarray,               # Complex array of e_phi values [freq, theta, phi]
    polarization: Optional[str] = None  # Optional polarization type
)
```

### Properties
- `frequencies` - Array of frequencies in Hz
- `theta_angles` - Array of theta angles in degrees
- `phi_angles` - Array of phi angles in degrees
- `polarization` - Current polarization type
- `data` - xarray.Dataset containing pattern data with coordinates and variables

### Key Methods
```python
# Pattern manipulation
pattern.assign_polarization(polarization: str) -> None
pattern.change_polarization(new_polarization: str) -> AntennaPattern
pattern.translate(translation: np.ndarray) -> AntennaPattern
pattern.swap_polarization_axes() -> AntennaPattern

# Phase center
pattern.find_phase_center(theta_angle: float, frequency: Optional[float] = None) -> np.ndarray
pattern.shift_to_phase_center(theta_angle: float, frequency: Optional[float] = None) -> Tuple[AntennaPattern, np.ndarray]

# Analysis
pattern.apply_mars(maximum_radial_extent: float) -> AntennaPattern
pattern.beamwidth_from_pattern(gain_pattern: np.ndarray, angles: np.ndarray, level_db: float = -3.0) -> float:
pattern.calculate_beamwidth(frequency: Optional[float] = None, level_db: float = -3.0) -> Dict[str, float]
pattern.get_gain_db(component: str = 'e_co') -> xr.DataArray
pattern.get_phase(component: str = 'e_co', unwrapped: bool = False) -> xr.DataArray
pattern.get_polarization_ratio() -> xr.DataArray
pattern.get_axial_ratio() -> xr.DataArray

# I/O
pattern.write_cut(file_path: Union[str, Path], polarization_format: int = 1) -> None

# Utility
pattern.clear_cache() -> None
pattern.at_frequency(frequency: float) -> ContextManager  # Context manager for single-frequency view
```

## Valid Polarization Values
- 'rhcp', 'rh', 'r' - Right-hand circular
- 'lhcp', 'lh', 'l' - Left-hand circular
- 'x', 'l3x' - Linear X (Ludwig's 3rd)
- 'y', 'l3y' - Linear Y (Ludwig's 3rd)
- 'theta' - Spherical theta
- 'phi' - Spherical phi

## File I/O Functions
```python
# Reading patterns
read_cut(file_path: Union[str, Path], frequency_start: float, frequency_end: float) -> AntennaPattern
read_ffd(file_path: Union[str, Path]) -> AntennaPattern

# Saving/loading NPZ
save_pattern_npz(pattern: AntennaPattern, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None
load_pattern_npz(file_path: Union[str, Path]) -> Tuple[AntennaPattern, Dict[str, Any]]
```

## Polarization Conversion Functions
```python
polarization_tp2xy(phi: RealArray, e_theta: ComplexArray, e_phi: ComplexArray) -> Tuple[ComplexArray, ComplexArray]
polarization_xy2pt(phi: RealArray, e_x: ComplexArray, e_y: ComplexArray) -> Tuple[ComplexArray, ComplexArray]
polarization_tp2rl(phi: RealArray, e_theta: ComplexArray, e_phi: ComplexArray) -> Tuple[ComplexArray, ComplexArray]
polarization_rl2xy(e_right: ComplexArray, e_left: ComplexArray) -> Tuple[ComplexArray, ComplexArray]
polarization_rl2tp(phi: RealArray, e_right: ComplexArray, e_left: ComplexArray) -> Tuple[ComplexArray, ComplexArray]
phase_pattern_translate(frequency, theta, phi, translation, phase_pattern) -> np.ndarray
```

## Analysis Functions
```python
find_phase_center(pattern, theta_angle: float, frequency: Optional[float] = None) -> np.ndarray
calculate_beamwidth(pattern, frequency: Optional[float] = None, level_db: float = -3.0) -> Dict[str, float]
apply_mars(pattern, maximum_radial_extent: float) -> AntennaPattern
translate_phase_pattern(pattern, translation) -> AntennaPattern
principal_plane_phase_center(frequency, theta1, theta2, theta3, phase1, phase2, phase3) -> Tuple[np.ndarray, np.ndarray]
get_axial_ratio(pattern) -> xr.DataArray
```

## Utility Functions
```python
frequency_to_wavelength(frequency: Union[float, np.ndarray], dielectric_constant: float = 1.0) -> np.ndarray
wavelength_to_frequency(wavelength: Union[float, np.ndarray], dielectric_constant: float = 1.0) -> np.ndarray
db_to_linear(db_value: Union[float, np.ndarray]) -> np.ndarray
linear_to_db(linear_value: Union[float, np.ndarray]) -> np.ndarray
beamwidth_from_pattern(gain_pattern: np.ndarray, angles: np.ndarray, level_db: float = -3.0) -> float
unwrap_phase(phase: np.ndarray, discont: float = np.pi) -> np.ndarray
```

## Key Dependencies
- numpy
- scipy
- xarray

## Import Examples
```python
# Import specific components
from antenna_pattern import AntennaPattern, read_cut, read_ffd
from antenna_pattern import find_phase_center, calculate_beamwidth, apply_mars
from antenna_pattern import polarization_tp2xy, polarization_tp2rl

# Or import everything
from antenna_pattern import *
```

## Basic Usage Examples
```python
# Load a pattern from a file
pattern = read_ffd("antenna.ffd")

# Create a pattern from data
pattern = AntennaPattern(
    theta=np.linspace(-180, 180, 361),
    phi=np.array([0, 45, 90, 135]),
    frequency=np.array([10e9]),  # 10 GHz
    e_theta=e_theta_data,
    e_phi=e_phi_data,
    polarization="theta"
)

# Get gain in dB for co-pol component
gain_db = pattern.get_gain_db('e_co')

# Calculate beamwidth
beamwidth = pattern.calculate_beamwidth(frequency=10e9, level_db=-3.0)
print(f"E-plane beamwidth: {beamwidth['E_plane']:.2f} degrees")

# Find phase center
phase_center = pattern.find_phase_center(theta_angle=30.0, frequency=10e9)

# Create a new pattern with a different polarization
rhcp_pattern = pattern.change_polarization("rhcp")

# Translate pattern to phase center
shifted_pattern, translation = pattern.shift_to_phase_center(theta_angle=30.0)

# Apply MARS algorithm with 0.5m maximum radial extent
cleaned_pattern = pattern.apply_mars(maximum_radial_extent=0.5)

# Save pattern to NPZ with metadata
save_pattern_npz(pattern, "antenna.npz", metadata={"source": "measurement"})
```