# AntennaPattern Package Reference

## Overview
AntennaPattern provides a class method for storing far field antenna patterns and performing antenna pattern analysis. It handles reading/writing patterns, polarization conversions, phase center analysis and manipulations, reflection supressions (using MARS) and various analysis methods.

## Package Structure
```
antenna_pattern/
├── __init__.py
├── pattern.py       # AntennaPattern class
├── ant_io.py        # File I/O functions
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
pattern.beamwidth_from_pattern(gain_pattern: np.ndarray, angles: np.ndarray, level_db: float = -3.0) -> float
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
find_nearest(array: np.ndarray, value: float) -> Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]
unwrap_phase(phase: np.ndarray, discont: float = np.pi) -> np.ndarray
interpolate_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float
```

## Key Dependencies
- numpy
- scipy
- xarray
- matplotlib (for visualizations)

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

### Loading and Creating Patterns
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
```

### Analysis
```python
# Get gain in dB for co-pol component
gain_db = pattern.get_gain_db('e_co')

# Calculate beamwidth
beamwidth = pattern.calculate_beamwidth(frequency=10e9, level_db=-3.0)
print(f"E-plane beamwidth: {beamwidth['E_plane']:.2f} degrees")
print(f"H-plane beamwidth: {beamwidth['H_plane']:.2f} degrees")
print(f"Average beamwidth: {beamwidth['Average']:.2f} degrees")

# Find phase center
phase_center = pattern.find_phase_center(theta_angle=30.0, frequency=10e9)
print(f"Phase center coordinates [x, y, z]: {phase_center}")
```

### Polarization Conversion
```python
# Create a new pattern with a different polarization
rhcp_pattern = pattern.change_polarization("rhcp")

# Get axial ratio of circularly polarized pattern
axial_ratio = rhcp_pattern.get_axial_ratio()
```

### Using the at_frequency Context Manager
```python
# Work with a single frequency from a multi-frequency pattern
with pattern.at_frequency(10e9) as single_freq_pattern:
    # All operations here will use only the 10 GHz slice
    beamwidth = single_freq_pattern.calculate_beamwidth(level_db=-3.0)
    phase_center = single_freq_pattern.find_phase_center(theta_angle=30.0)
    
    # Data manipulations only affect this view, not the original pattern
    shifted_pattern = single_freq_pattern.translate([0, 0, 0.1])
```

### Method Chaining
```python
# Methods can be chained for a more streamlined workflow
result_pattern = (
    pattern
    .change_polarization("rhcp")
    .translate([0, 0, 0.1])
    .apply_mars(maximum_radial_extent=0.5)
)
```


### Visualization Example
```python
import matplotlib.pyplot as plt
import numpy as np

# Plot gain patterns in principal planes
def plot_pattern_cuts(pattern, frequency=None, level_db=-3.0):
    # If frequency provided, use at_frequency context manager
    if frequency is not None:
        with pattern.at_frequency(frequency) as single_freq_pattern:
            return plot_pattern_cuts(single_freq_pattern)
    
    # Get co-polarized gain
    gain_db = pattern.get_gain_db('e_co')
    
    # Get indices for principal planes
    phi_0_idx = np.argmin(np.abs(pattern.phi_angles - 0))
    phi_90_idx = np.argmin(np.abs(pattern.phi_angles - 90))
    
    # Extract the cuts (use first frequency)
    freq_idx = 0
    e_plane = gain_db[freq_idx, :, phi_0_idx].values
    h_plane = gain_db[freq_idx, :, phi_90_idx].values
    
    # Make the plot
    plt.figure(figsize=(10, 6))
    plt.plot(pattern.theta_angles, e_plane, 'b-', label='E-Plane (φ=0°)')
    plt.plot(pattern.theta_angles, h_plane, 'r-', label='H-Plane (φ=90°)')
    
    # Calculate beamwidth
    beamwidth = pattern.calculate_beamwidth(level_db=level_db)
    
    # Plot beamwidth level
    max_gain = np.max([np.max(e_plane), np.max(h_plane)])
    bw_level = max_gain + level_db
    plt.axhline(y=bw_level, color='k', linestyle='--', 
                label=f'{-level_db} dB Beamwidth Level')
    
    plt.xlabel('Theta (deg)')
    plt.ylabel('Gain (dB)')
    plt.title(f'Antenna Pattern at {pattern.frequencies[freq_idx]/1e9:.2f} GHz')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Add beamwidth annotation
    plt.annotate(f'E-Plane BW: {beamwidth["E_plane"]:.1f}°\n'
                 f'H-Plane BW: {beamwidth["H_plane"]:.1f}°', 
                 xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    return plt.gcf()

# Example call
fig = plot_pattern_cuts(pattern, frequency=10e9)
plt.show()
```

## Common Pitfalls and Tips

1. **Units**: Always use consistent units:
   - Frequencies in Hz (not MHz or GHz)
   - Angles in degrees (theta range -180 to 180, phi range 0 to 360)
   - Distances in meters

2. **Coordinate Systems**: 
   - The package uses a right-handed coordinate system
   - Z-axis is along boresight
   - Theta is the angle from the z-axis (0° at boresight)
   - Phi is the antenna head roll angle (0° along x-axis)