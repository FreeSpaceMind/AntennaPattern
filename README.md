# Antenna Pattern Library

A comprehensive toolkit for antenna radiation pattern analysis including:

- Reading and writing antenna patterns from various file formats
- Polarization conversions between spherical, Ludwig-3, and circular coordinates
- Phase center calculations and phase shifting
- Pattern analysis tools (beamwidths, axial ratio, etc.)
- MARS (Mathematical Absorber Reflection Suppression) algorithm

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

## Usage Example

```python
import numpy as np
from antenna_pattern import AntennaPattern, read_cut, read_ffd

# Load a pattern from a file
pattern = read_ffd("my_pattern.ffd")

# Or create a pattern with synthetic data
theta = np.linspace(-180, 180, 361)
phi = np.array([0, 45, 90, 135])
frequency = np.array([10e9])  # 10 GHz

# Create sample field components
e_theta = np.zeros((1, len(theta), len(phi)), dtype=complex)
e_phi = np.zeros((1, len(theta), len(phi)), dtype=complex)

# Fill with a simple analytical pattern
for p_idx, p_val in enumerate(phi):
    e_theta[0, :, p_idx] = np.cos(np.radians(theta/2)) ** 2
    e_phi[0, :, p_idx] = np.sin(np.radians(theta/2)) * np.sin(np.radians(p_val))

# Create antenna pattern
pattern = AntennaPattern(
    theta=theta,
    phi=phi,
    frequency=frequency,
    e_theta=e_theta,
    e_phi=e_phi,
    polarization="theta"
)

# Calculate beamwidth at 3dB level
beamwidth = pattern.calculate_beamwidth(frequency=10e9, level_db=-3.0)
print(f"E-plane beamwidth: {beamwidth['E_plane']:.2f} degrees")
print(f"H-plane beamwidth: {beamwidth['H_plane']:.2f} degrees")

# Find phase center
phase_center = pattern.find_phase_center(theta_angle=30.0, frequency=10e9)
print(f"Phase center: {phase_center} meters")

# Create a new pattern with shifts
shifted_pattern = pattern.translate(phase_center)
```

## Features

### Importing, Converting, Exporting Far-Field Patterns
The library allows importing, conversion between, and exporting of common antenna files:
- **.ffd** : this is the HFSS far field data format.
- **.cut** : this is the GRASP cut file format. .cut files do not natively include frequenc, so that must be provided as well.
- **.npz** : a numpy zip file used for saving and loading patterns within this library. Faster than reading .ffd or .cut repeatedly.

### Polarization Conversions

The library supports conversions between:
- Spherical (θ, φ)
- Ludwig-3 (x, y)
- Circular (RHCP, LHCP)

### Phase Center Analysis

This library allows manipulations of the phase origin reference so that phase center analysis can be done. The library includes an optimizer to find the optimum phase center within a beamwidth.

```python
phase_center = pattern.find_phase_center(theta_angle=30.0)
shifted_pattern = pattern.translate(phase_center)
```

### MARS Algorithm

Apply the Mathematical Absorber Reflection Suppression (MARS) algorithm to mitigate chamber reflections, see ["Application of Mathematical Absorber Reflection Supression
to Direct Far-Field Antenna Range Measurements](https://www.nsi-mi.com/-/media/project/oneweb/oneweb/nsi/files/technical-papers/2011/application-of-mathematical-absorber-reflection-suppression-to-direct-far-field-antenna-range-measurements.pdf?la=en&revision=0a4b7b72-f427-4e4f-992c-40c0377fff2a&hash=6CBFD6C61EFAB9AA82ED1443A6C2F89C). Note that effective use of MARS requires that the antenna was displaced from the center of rotation during measurement. The antenna must be shifted back to the center of rotation, and then MARS can be applied.

```python
# Find the antenna phase center and move the phase pattern to that location
phase_center = pattern.find_phase_center(theta_angle=30.0)
shifted_pattern = pattern.translate(phase_center)

# Apply MARS with a maximum radial extent of 0.5 meters
clean_pattern = shited_pattern.apply_mars(maximum_radial_extent=0.5)
```

## Usage with AI
The AntennaPattern_reference.md file is intended to be a minimal instruction set to provide to AI models so that they can sucessfully use the AntennaPattern package.

## License

This project is licensed under the MIT License - see the LICENSE file for details.