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

### Polarization Conversions

The library supports conversions between:
- Spherical (θ, φ)
- Ludwig-3 (x, y)
- Circular (RHCP, LHCP)

### Phase Center Analysis

Find optimal phase centers to minimize phase variations across the beam:

```python
phase_center = pattern.find_phase_center(theta_angle=30.0)
shifted_pattern = pattern.translate(phase_center)
```

### MARS Algorithm

Apply the Mathematical Absorber Reflection Suppression algorithm to mitigate chamber reflections:

```python
# Apply MARS with a maximum radial extent of 0.5 meters
clean_pattern = pattern.apply_mars(maximum_radial_extent=0.5)
```

## Usage with AI
The AntennaPattern_reference.md file is intended to be a minimal instruction set to provide to AI models so that they can sucessfully use the AntennaPattern package.

digest.txt is a full set of the codebase generated with gitingest, and can be provided if the AI tool requires more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.