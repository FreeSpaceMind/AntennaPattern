# Antenna Pattern Library

A comprehensive toolkit for antenna radiation pattern analysis including:

- Reading and writing antenna patterns from various file formats
- Polarization conversions between spherical, Ludwig-3, and circular coordinates
- Phase center calculations and phase shifting
- Pattern analysis tools (beamwidths, axial ratio, etc.)
- MARS (Mathematical Absorber Reflection Suppression) algorithm
- Synthetic pattern generation

## Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install antenna_pattern
```

### Option 2: Install from source

#### Basic installation
```bash
git clone https://github.com/freespacemind/antenna_pattern.git
cd antenna_pattern
pip install -e .
```

#### Using the installation script
```bash
git clone https://github.com/freespacemind/antenna_pattern.git
cd antenna_pattern
python install.py
```

### Option 3: Direct installation from GitHub
```bash
pip install git+https://github.com/freespacemind/antenna_pattern.git
```

## Development Setup

To set up a development environment:

```bash
# Clone repository
git clone https://github.com/freespacemind/antenna_pattern.git
cd antenna_pattern

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Basic Usage

### Creating a Synthetic Pattern
The synthetic pattern creating function is included mostly for the purpose of testing the package. In most cases, you should be importing a pattern from elsewhere.

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
corrected_pattern, translation = pattern.shift_to_phase_center(theta_angle=30.0)
```

### Scaling Pattern Gain

```python
# Uniform scaling (apply same scale to all frequencies)
scaled_pattern = pattern.scale_pattern(4.0)  # Add 4 dB to all angles

# Frequency-dependent scaling
freq_scale = np.array([2.0, 4.0, 6.0])  # Different scaling for each frequency
freq_scaled_pattern = pattern.scale_pattern(freq_scale)
```

### Saving Patterns

```python
from antenna_pattern.ant_io import save_pattern_npz

# Save to NPZ format with metadata
metadata = {"description": "My pattern", "created_by": "me"}
save_pattern_npz(pattern, "my_pattern.npz", metadata)

# Save to GRASP CUT format with different polarization formats:
# 1: theta/phi, 2: RHCP/LHCP, 3: Ludwig-3 (x/y)
pattern.write_cut("my_pattern_tp.cut", polarization_format=1)
pattern.write_cut("my_pattern_rl.cut", polarization_format=2)
pattern.write_cut("my_pattern_xy.cut", polarization_format=3)
```

## Features

### Importing, Converting, Exporting Far-Field Patterns
The library allows importing, conversion between, and exporting of common antenna files:
- **.ffd** : HFSS far field data format
- **.cut** : GRASP cut file format. .cut files do not natively include frequency, so that must be provided as well
- **.npz** : numpy zip file used for saving and loading patterns within this library. Faster than reading .ffd or .cut repeatedly

### Polarization Conversions

The library supports conversions between:
- Spherical (θ, φ)
- Ludwig-3 (x, y)
- Circular (RHCP, LHCP)

### Phase Center Analysis

This library allows manipulation of the phase origin reference so that phase center analysis can be done. The library includes an optimizer to find the optimum phase center within a beamwidth.

```python
phase_center = pattern.find_phase_center(theta_angle=30.0)
shifted_pattern = pattern.translate(phase_center)
```

### MARS Algorithm

Apply the Mathematical Absorber Reflection Suppression (MARS) algorithm to mitigate chamber reflections, see ["Application of Mathematical Absorber Reflection Suppression
to Direct Far-Field Antenna Range Measurements](https://www.nsi-mi.com/-/media/project/oneweb/oneweb/nsi/files/technical-papers/2011/application-of-mathematical-absorber-reflection-suppression-to-direct-far-field-antenna-range-measurements.pdf?la=en&revision=0a4b7b72-f427-4e4f-992c-40c0377fff2a&hash=6CBFD6C61EFAB9AA82ED1443A6C2F89C). Note that effective use of MARS requires that the antenna was displaced from the center of rotation during measurement. The antenna must be shifted back to the center of rotation, and then MARS can be applied.

```python
# Find the antenna phase center and move the phase pattern to that location
phase_center = pattern.find_phase_center(theta_angle=30.0)
shifted_pattern = pattern.translate(phase_center)

# Apply MARS with a maximum radial extent of 0.5 meters
clean_pattern = shifted_pattern.apply_mars(maximum_radial_extent=0.5)
```

### Synthetic Pattern Generation

Create antenna patterns from high-level parameters without requiring detailed electromagnetic modeling:

```python
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
```

## Example Files

See the `example_tests` directory for detailed examples of using the library, including:

- Creating synthetic patterns
- Saving and loading patterns
- Converting between polarization types
- Phase center analysis and MARS algorithm
- Pattern gain scaling

## Usage with AI
The AntennaPattern_reference.md file is intended to be a minimal instruction set to provide to AI models so that they can successfully use the AntennaPattern package.

## License

This project is licensed under the MIT License - see the LICENSE file for details.