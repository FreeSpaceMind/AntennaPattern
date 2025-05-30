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
    polarization_xy2tp,
    polarization_tp2rl,
    polarization_rl2xy,
    polarization_rl2tp
)
from .pattern_functions import (
    unwrap_phase,
    normalize_phase,
    translate_phase_pattern,
    phase_pattern_translate,
    scale_amplitude,
    transform_tp2uvw,
    transform_uvw2tp,
    isometric_rotation,
    mirror_pattern,
    normalize_at_boresight,
    shift_theta_origin,
    shift_phi_origin
)
from .analysis import (
    calculate_phase_center,
    principal_plane_phase_center,
    get_axial_ratio,
    get_phase_length,
    get_group_delay
)
from .utilities import (
    find_nearest,
    frequency_to_wavelength,
    wavelength_to_frequency,
    lightspeed,
    freespace_permittivity,
    freespace_impedance,
    db_to_linear,
    linear_to_db,
    interpolate_crossing,
    create_synthetic_pattern
)
from .plotting import (
    plot_pattern_cut,
    plot_multiple_patterns,
    plot_pattern_difference,
    plot_pattern_statistics,
    add_spec_mask,
    add_envelope_spec,
    plot_phase_delay_pattern,
    plot_phase_slope_vs_frequency
)

from .package_functions import (
    average_patterns,
    difference_patterns
)

# Define what gets imported with "from antenna_pattern import *"
__all__ = [
    'AntennaPattern',
    'read_cut',
    'read_ffd',
    'load_pattern_npz',
    'save_pattern_npz',
    'polarization_tp2xy',
    'polarization_xy2tp',
    'polarization_tp2rl',
    'polarization_rl2xy',
    'polarization_rl2tp',
    'unwrap_phase',
    'normalize_phase',
    'translate_phase_pattern',
    'phase_pattern_translate',
    'scale_amplitude',
    'transform_tp2uvw',
    'transform_uvw2tp',
    'isometric_rotation',
    'mirror_pattern',
    'normalize_at_boresight',
    'shift_theta_origin',
    'shift_phi_origin',
    'calculate_phase_center',
    'principal_plane_phase_center',
    'get_axial_ratio',
    'get_phase_length',
    'get_group_delay',
    'find_nearest',
    'frequency_to_wavelength',
    'wavelength_to_frequency',
    'lightspeed',
    'freespace_permittivity',
    'freespace_impedance',
    'db_to_linear',
    'linear_to_db',
    'interpolate_crossing',
    'create_synthetic_pattern',
    'plot_pattern_cut',
    'plot_multiple_patterns',
    'plot_pattern_difference',
    'plot_pattern_statistics',
    'add_spec_mask',
    'add_envelope_spec',
    'plot_phase_delay_pattern',
    'plot_phase_slope_vs_frequency',
    'average_patterns',
    'difference_patterns'
]