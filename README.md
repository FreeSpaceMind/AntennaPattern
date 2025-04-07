# Antenna Pattern Library

A comprehensive toolkit for antenna radiation pattern analysis including:

- Reading and writing antenna patterns from various file formats
- Polarization conversions between spherical, Ludwig-3, and circular coordinates
- Phase center calculations and phase shifting
- Pattern analysis tools (beamwidths, axial ratio, etc.)
- MARS (Mathematical Absorber Reflection Suppression) algorithm
- Synthetic pattern generation
- Coordinate transformations and visualization tools

## Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install antenna_pattern
```

### Option 2: Install from source

```bash
git clone https://github.com/freespacemind/antenna_pattern.git
cd antenna_pattern
pip install -e .
```

### Option 3: Using the installation script

```bash
git clone https://github.com/freespacemind/antenna_pattern.git
cd antenna_pattern
python install.py
```

### Option 4: Direct installation from GitHub

```bash
pip install git+https://github.com/freespacemind/antenna_pattern.git
```

## Development Setup

To set up a development environment:

```bash
# Clone repository
git clone https://github.com/freespacemind/antenna_pattern.git
cd antenna_pattern

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Package Structure

```
antenna_pattern/
├── __init__.py       # Package exports
├── pattern.py        # AntennaPattern class
├── ant_io.py         # File I/O functions
├── polarization.py   # Polarization conversions
├── pattern_functions.py # Core pattern operations
├── analysis.py       # Analysis functions
├── utilities.py      # Common utilities
└── plotting.py       # Visualization tools
```

## Basic Usage

### Loading a Pattern from a File

```python
from antenna_pattern import read_ffd, read_cut, load_pattern_npz

# Load from HFSS .ffd format
pattern_ffd = read_ffd("my_pattern.ffd")

# Load from GRASP .cut format (need to specify frequency range)
pattern_cut = read_cut("my_pattern.cut", 8e9, 12e9)

# Load from .npz format (fastest)
pattern_npz, metadata = load_pattern_npz("my_pattern.npz")
```

### Working with Pattern Data

```python
# Get basic information
print(f"Frequencies: {pattern.frequencies / 1e9} GHz")
print(f"Theta range: {min(pattern.theta_angles)}° to {max(pattern.theta_angles)}°")
print(f"Phi range: {min(pattern.phi_angles)}° to {max(pattern.phi_angles)}°")
print(f"Polarization: {pattern.polarization}")

# Access pattern data using xarray interface
e_theta_data = pattern.data.e_theta
e_phi_data = pattern.data.e_phi
e_co_data = pattern.data.e_co
e_cx_data = pattern.data.e_cx

# Extract single-frequency pattern
with pattern.at_frequency(10e9) as single_freq_pattern:
    # Work with single_freq_pattern
    gain_db = single_freq_pattern.get_gain_db('e_co')
```

### Changing Polarization

```python
# Convert pattern to different polarization representations
rhcp_pattern = pattern.change_polarization("rhcp")
lhcp_pattern = pattern.change_polarization("lhcp")
linear_x_pattern = pattern.change_polarization("x")
linear_y_pattern = pattern.change_polarization("y")
theta_pattern = pattern.change_polarization("theta")
phi_pattern = pattern.change_polarization("phi")
```

### Accessing Gain and Phase Data

```python
# Get the gain in dB for different components
co_pol_gain_db = pattern.get_gain_db('e_co')
cross_pol_gain_db = pattern.get_gain_db('e_cx')
theta_gain_db = pattern.get_gain_db('e_theta')
phi_gain_db = pattern.get_gain_db('e_phi')

# Get the phase in radians
co_pol_phase = pattern.get_phase('e_co')
co_pol_phase_unwrapped = pattern.get_phase('e_co', unwrapped=True)
```

### Pattern Analysis

```python
# Calculate beamwidth at specified level
beamwidth = pattern.calculate_beamwidth(frequency=10e9, level_db=-3.0)
print(f"E-plane beamwidth: {beamwidth['E_plane']:.2f} degrees")
print(f"H-plane beamwidth: {beamwidth['H_plane']:.2f} degrees")

# Get axial ratio
axial_ratio = pattern.get_axial_ratio()

# Find phase center
phase_center = pattern.find_phase_center(theta_angle=30.0, frequency=10e9)
print(f"Phase center: {phase_center} meters")

# Shift pattern to phase center
pattern.shift_to_phase_center(theta_angle=30.0)
```

### Coordinate Transformations

```python
# Convert between spherical coordinates and direction cosines
from antenna_pattern import transform_tp2uvw, transform_uvw2tp

# Convert theta/phi to direction cosines
u, v, w = transform_tp2uvw(theta, phi)

# Convert direction cosines back to theta/phi
theta, phi = transform_uvw2tp(u, v, w)

# Apply a rotation to the pattern
from antenna_pattern import isometric_rotation

# Rotate direction cosines by specified angles (azimuth, elevation, roll)
u_rot, v_rot, w_rot = isometric_rotation(u, v, w, azimuth=45.0, elevation=10.0, roll=0.0)
```

### Plotting Patterns

```python
from antenna_pattern import plot_pattern_cut
import matplotlib.pyplot as plt

# Plot standard gain pattern
fig = plot_pattern_cut(
    pattern,
    frequency=10e9,
    phi=[0, 90],
    show_cross_pol=True,
    value_type='gain'
)

# Plot phase pattern
phase_fig = plot_pattern_cut(
    pattern,
    frequency=10e9,
    phi=0,
    value_type='phase',
    unwrap_phase=True
)

# Plot axial ratio
ar_fig = plot_pattern_cut(
    pattern,
    frequency=10e9,
    phi=[0, 90],
    value_type='axial_ratio'
)

plt.show()
```

### Creating Synthetic Patterns

```python
from antenna_pattern import AntennaPattern, create_synthetic_pattern
import numpy as np

# Define pattern parameters
frequencies = np.array([10e9])  # 10 GHz
theta = np.linspace(-90, 90, 181)  # 1-degree steps
phi = np.array([0, 90])  # Principal planes

# Create synthetic pattern with specified parameters
e_theta, e_phi = create_synthetic_pattern(
    frequencies=frequencies,
    theta_angles=theta,
    phi_angles=phi,
    peak_gain_dbi=15.0,           # Peak gain in dBi
    polarization='rhcp',          # Circular polarization
    beamwidth_deg=30.0,           # 3dB beamwidth
    axial_ratio_db=1.0,           # Axial ratio in dB
    front_to_back_db=25.0,        # Front-to-back ratio
    sidelobe_level_db=-20.0       # Sidelobe level
)

# Create AntennaPattern object
pattern = AntennaPattern(
    theta=theta,
    phi=phi,
    frequency=frequencies,
    e_theta=e_theta,
    e_phi=e_phi
)
```

### Saving Patterns

```python
from antenna_pattern import save_pattern_npz

# Save to NPZ format with metadata
metadata = {"description": "My pattern", "created_by": "me"}
save_pattern_npz(pattern, "my_pattern.npz", metadata)

# Save to GRASP CUT format with different polarization formats:
# 1: theta/phi, 2: RHCP/LHCP, 3: Ludwig-3 (x/y)
pattern.write_cut("my_pattern_tp.cut", polarization_format=1)
pattern.write_cut("my_pattern_rl.cut", polarization_format=2)
pattern.write_cut("my_pattern_xy.cut", polarization_format=3)
```

### MARS (Mathematical Absorber Reflection Suppression)

```python
# Apply the MARS algorithm to mitigate chamber reflections
# Note: effective use of MARS requires that the antenna was displaced from
# the center of rotation during measurement

# Find the antenna phase center and move the phase pattern to that location
phase_center = pattern.find_phase_center(theta_angle=30.0)
pattern.translate(phase_center)

# Apply MARS with a maximum radial extent of 0.5 meters
pattern.apply_mars(maximum_radial_extent=0.5)
```

### Pattern Scaling

```python
# Uniform scaling (apply same scale to all frequencies)
pattern.scale_pattern(4.0)  # Add 4 dB to all angles

# Frequency-dependent scaling
freq_scale = np.array([2.0, 4.0, 6.0])  # Different scaling for each frequency
pattern.scale_pattern(freq_scale)

# 2D scaling grid (frequency and phi-dependent)
freq_scale = np.array([8e9, 10e9, 12e9])
phi_scale = np.array([0, 45, 90, 135])
scale_2d = np.zeros((len(freq_scale), len(phi_scale)))

# Fill in scaling values
scale_2d[0, :] = [1.0, 2.0, 3.0, 4.0]  # 8 GHz scaling vs phi
scale_2d[1, :] = [2.0, 3.0, 4.0, 5.0]  # 10 GHz scaling vs phi
scale_2d[2, :] = [3.0, 4.0, 5.0, 6.0]  # 12 GHz scaling vs phi

# Apply 2D scaling grid
pattern.scale_pattern(scale_2d, freq_scale=freq_scale, phi_scale=phi_scale)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.