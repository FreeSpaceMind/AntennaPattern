"""
AntennaPattern package - Core functionality for antenna pattern analysis.

This package provides the AntennaPattern class and related functions for
working with antenna radiation patterns, including reading/writing patterns,
polarization conversions, and analysis tools.
"""

__version__ = '0.1.0'
__author__ = 'Justin Long'
__email__ = 'justinwlong1@gmail.com'

# Import key classes and functions to make them available at the package level
from .pattern import AntennaPattern
from .ant_io import read_cut, read_ffd, load_pattern_npz, save_pattern_npz
from .polarization import (
    polarization_tp2xy,
    polarization_xy2pt,
    polarization_tp2rl,
    polarization_rl2xy,
    polarization_rl2tp
)
from .analysis import (
    find_phase_center,
    translate_phase_pattern,
    beamwidth_from_pattern,
    calculate_beamwidth,
    apply_mars,
    get_axial_ratio,
    normalize_phase
)
from .utilities import (
    find_nearest,
    frequency_to_wavelength,
    lightspeed,
    freespace_permittivity,
    freespace_impedance,
    db_to_linear,
    linear_to_db,
    scale_amplitude,
    create_synthetic_pattern,
    transform_tp2uvw,
    transform_uvw2tp,
    isometric_rotation,
)
from .plotting import (
    plot_pattern_cut,
)

# Define what gets imported with "from antenna_pattern import *"
__all__ = [
    'AntennaPattern',
    'read_cut',
    'read_ffd',
    'load_pattern_npz',
    'save_pattern_npz',
    'polarization_tp2xy',
    'polarization_xy2pt',
    'polarization_tp2rl',
    'polarization_rl2xy',
    'polarization_rl2tp',
    'find_phase_center',
    'translate_phase_pattern',
    'beamwidth_from_pattern',
    'calculate_beamwidth',
    'apply_mars',
    'get_axial_ratio',
    'normalize_phase',
    'find_nearest',
    'frequency_to_wavelength',
    'lightspeed',
    'freespace_permittivity',
    'freespace_impedance',
    'db_to_linear',
    'linear_to_db',
    'scale_amplitude',
    'create_synthetic_pattern',
    'transform_tp2uvw',
    'transform_uvw2tp',
    'isometric_rotation',
    'plot_pattern_cut',
]