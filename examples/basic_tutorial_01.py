#!/usr/bin/env python3
"""
Basic Antenna Pattern Loading and Visualization Tutorial

This tutorial demonstrates the fundamental operations for working with antenna
patterns using the AntennaPattern package:
- Loading patterns from common file formats
- Basic pattern visualization
- Simple pattern analysis
- Exporting to different formats

This is a beginner-friendly introduction that doesn't require complex measurement
processing or advanced antenna theory knowledge.
"""

from antenna_pattern import (
    read_cut, read_ffd, load_pattern_npz, save_pattern_npz,
    plot_pattern_cut, plot_multiple_patterns,
    create_synthetic_pattern, AntennaPattern
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)

def print_pattern_info(pattern, name="Pattern"):
    """Print basic information about a pattern."""
    print(f"\n{name} Information:")
    print(f"  Frequencies: {len(pattern.frequencies)} points from {pattern.frequencies[0]/1e6:.1f} to {pattern.frequencies[-1]/1e6:.1f} MHz")
    print(f"  Theta range: {pattern.theta_angles.min():.1f}° to {pattern.theta_angles.max():.1f}°")
    print(f"  Phi range: {pattern.phi_angles.min():.1f}° to {pattern.phi_angles.max():.1f}°")
    print(f"  Polarization: {pattern.polarization}")
    
    # Calculate some basic metrics
    max_gain = np.max(pattern.get_gain_db('e_co').values)
    min_gain = np.min(pattern.get_gain_db('e_co').values)
    print(f"  Peak gain: {max_gain:.2f} dBi")
    print(f"  Min gain: {min_gain:.2f} dBi")
    print(f"  Dynamic range: {max_gain - min_gain:.1f} dB")

# Get the directory where this script is located
script_dir = Path(__file__).parent

# ============================================================================
print_section_header("TUTORIAL 1: LOADING ANTENNA PATTERN FROM FFD FILE")
# ============================================================================
print("""
This tutorial shows importing an FFD filr format, which is native to HFSS.
This file is created in the complex workflow tutorial from measured data imported as
a .CUT file. FFD files are convenient in that they include the frequency information,
unlike CUT files.
""")

# Load the processed pattern from FFD file
ffd_file = script_dir / 'data' / 'sample_pattern_simple.ffd'

try:
    main_pattern = read_ffd(ffd_file)
    print("✓ Successfully loaded .FFD file")
    print_pattern_info(main_pattern, "Processed Antenna Pattern")
except Exception as e:
    print(f"✗ Error loading .FFD file: {e}")
    sys.exit(1)

# ============================================================================
print_section_header("TUTORIAL 2: BASIC PATTERN VISUALIZATION")
# ============================================================================
print("""
Now let's visualize the antenna pattern using different plot types.
The most common plots are:
- Gain patterns (co-pol and cross-pol)
- Phase patterns
- Axial ratio patterns (for circularly polarized antennas)

We'll plot these at a single frequency to start.
""")

# Select analysis frequency
analysis_freq = 3.2e9
print(f"Analysis frequency: {analysis_freq/1e6:.1f} MHz")

# Plot 1: Gain pattern with both co-pol and cross-pol
print("\nPlotting gain pattern...")
fig1 = plot_pattern_cut(
    main_pattern,
    frequency=analysis_freq,
    show_cross_pol=True,  # Show both co-pol and cross-pol
    title=f'Antenna Gain Pattern - {analysis_freq/1e6:.1f} MHz'
)
plt.xlabel('Theta (degrees)')
plt.ylabel('Gain (dBi)')
plt.grid(True)
plt.xlim([-90, 90])  # Focus on main beam region
plt.ylim([-30, 10])
plt.show()

# Plot 2: Phase pattern (co-pol only)
print("Plotting phase pattern...")
fig2 = plot_pattern_cut(
    main_pattern,
    frequency=analysis_freq,
    value_type='phase',
    show_cross_pol=False,  # Phase plots usually show co-pol only
    unwrap_phase=True,     # Remove 2π phase jumps
    title=f'Antenna Phase Pattern - {analysis_freq/1e6:.1f} MHz'
)
plt.xlabel('Theta (degrees)')
plt.ylabel('Phase (degrees)')
plt.grid(True)
plt.xlim([-90, 90])
plt.ylim([-10, 10])
plt.show()

# Plot 3: Axial ratio (for circularly polarized antennas)
if main_pattern.polarization.lower() in ['rhcp', 'lhcp', 'r', 'l']:
    print("Plotting axial ratio pattern...")
    fig3 = plot_pattern_cut(
        main_pattern,
        frequency=analysis_freq,
        value_type='axial_ratio',
        title=f'Axial Ratio Pattern - {analysis_freq/1e6:.1f} MHz'
    )
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Axial Ratio (dB)')
    plt.grid(True)
    plt.xlim([-90, 90])
    plt.ylim([0, 10])  # Typical AR range
    plt.show()

# ============================================================================
print_section_header("TUTORIAL 3: PATTERN ANALYSIS")
# ============================================================================
print("""
Let's perform some basic analysis on the antenna pattern to extract
key performance metrics that are commonly reported in antenna specifications.
""")

# Get pattern data for analysis
gain_data = main_pattern.get_gain_db('e_co')
phase_data = main_pattern.get_phase('e_co', unwrapped=True)

# Find frequency index for analysis
freq_idx = np.argmin(np.abs(main_pattern.frequencies - analysis_freq))

# Analysis at boresight (theta = 0)
theta_angles = main_pattern.theta_angles
boresight_idx = np.argmin(np.abs(theta_angles))
print(f"Boresight index: {boresight_idx} (theta = {theta_angles[boresight_idx]:.1f}°)")

# Peak gain analysis
peak_gain = np.max(gain_data.values[freq_idx])
peak_indices = np.unravel_index(
    np.argmax(gain_data.values[freq_idx]), 
    gain_data.values[freq_idx].shape
)
peak_theta = theta_angles[peak_indices[0]]
peak_phi = main_pattern.phi_angles[peak_indices[1]]

print(f"\nGain Analysis:")
print(f"  Peak gain: {peak_gain:.2f} dBi")
print(f"  Peak location: theta={peak_theta:.1f}°, phi={peak_phi:.1f}°")

# Beamwidth calculation (simple version for first phi cut)
phi_idx = 0  # Use first phi cut
gain_cut = gain_data.values[freq_idx, :, phi_idx]

# Find -3dB points
gain_3db = peak_gain - 3.0
indices_above_3db = np.where(gain_cut >= gain_3db)[0]

if len(indices_above_3db) > 0:
    theta_3db_low = theta_angles[indices_above_3db[0]]
    theta_3db_high = theta_angles[indices_above_3db[-1]]
    beamwidth_3db = theta_3db_high - theta_3db_low
    
    print(f"  -3dB beamwidth: {beamwidth_3db:.1f}° (phi={main_pattern.phi_angles[phi_idx]:.1f}° cut)")
    print(f"  -3dB points: {theta_3db_low:.1f}° to {theta_3db_high:.1f}°")
else:
    print("  Could not determine -3dB beamwidth")

# Cross-polarization analysis
if main_pattern.get_gain_db('e_cx') is not None:
    cx_gain_data = main_pattern.get_gain_db('e_cx')
    peak_cx_gain = np.max(cx_gain_data.values[freq_idx])
    cross_pol_isolation = peak_gain - peak_cx_gain
    print(f"  Cross-pol isolation: {cross_pol_isolation:.1f} dB")

# ============================================================================
print_section_header("TUTORIAL 4: COMPARING MULTIPLE PATTERNS")
# ============================================================================
print("""
Often we need to compare patterns at different frequencies, different phi cuts,
or entirely different antennas. Here is a quick example of plot_multiple_patterns,
which allows quick comparisons.
""")

# If we have multiple frequencies, compare them
if len(main_pattern.frequencies) > 1:
    print("Comparing patterns at different frequencies...")
    
    # Select 2-3 frequencies for comparison
    freq_indices = [0, len(main_pattern.frequencies)//2, -1]  # First, middle, last
    selected_freqs = [main_pattern.frequencies[i] for i in freq_indices]
    
    fig4 = plot_multiple_patterns(
        [main_pattern] * len(selected_freqs),  # Same pattern, different frequencies
        frequencies=selected_freqs,
        labels=[f'{f/1e6:.1f} MHz' for f in selected_freqs],
        title='Gain vs Frequency Comparison',
        show_cross_pol=False
    )
    plt.xlim([-90, 90])
    plt.ylim([-30, 10])
    plt.grid(True)
    plt.show()

# ============================================================================
print_section_header("TUTORIAL 5: POLARIZATION CONVERSION")
# ============================================================================
print("""
The AntennaPattern package can convert between different polarization
representations. Here is a quick example, converting from circular to linear.
""")

print(f"Current polarization: {main_pattern.polarization}")

# Create a copy and convert to different polarization
new_pol = 'x'

print(f"Converting to: {new_pol}")

# Make a copy and convert polarization
pattern_converted = AntennaPattern(
    theta=main_pattern.theta_angles,
    phi=main_pattern.phi_angles,
    frequency=main_pattern.frequencies,
    e_theta=main_pattern.data.e_theta.values.copy(),
    e_phi=main_pattern.data.e_phi.values.copy()
)

pattern_converted.change_polarization(new_pol)

print(f"✓ Conversion complete. New polarization: {pattern_converted.polarization}")

# Compare original and converted patterns
fig6 = plot_multiple_patterns(
    [main_pattern, pattern_converted],
    frequencies=[analysis_freq, analysis_freq],
    labels=[f'Original ({main_pattern.polarization})', f'Converted ({new_pol})'],
    title=f'Polarization Comparison - {analysis_freq/1e6:.1f} MHz'
)
plt.xlim([-90, 90])
plt.ylim([-30, 10])
plt.grid(True)
plt.show()

# ============================================================================
print_section_header("TUTORIAL 6: SAVING PATTERNS")
# ============================================================================
print("""
After loading and potentially modifying patterns, you'll want to save them
for future use. The AntennaPattern package supports several export formats.
""")

# Create output directory if it doesn't exist
output_dir = script_dir / 'output'
output_dir.mkdir(exist_ok=True)

print(f"Saving patterns to: {output_dir}")

# Save in native NPZ format (fastest loading)
npz_file = output_dir / 'tutorial_pattern.npz'
try:
    save_pattern_npz(
        main_pattern, 
        npz_file, 
        metadata={
            'description': 'Pattern from basic tutorial',
            'analysis_frequency': analysis_freq,
            'tutorial': 'basic_loading_visualization'
        }
    )
    print(f"✓ Saved NPZ file: {npz_file}")
except Exception as e:
    print(f"✗ Error saving NPZ: {e}")

# Save in FFD format (HFSS compatible)
ffd_file = output_dir / 'tutorial_pattern.ffd'
try:
    main_pattern.write_ffd(ffd_file)
    print(f"✓ Saved FFD file: {ffd_file}")
except Exception as e:
    print(f"✗ Error saving FFD: {e}")

# Save in CUT format (GRASP compatible)
cut_file = output_dir / 'tutorial_pattern.cut'
try:
    main_pattern.write_cut(cut_file, polarization_format=1)  # theta/phi format
    print(f"✓ Saved CUT file: {cut_file}")
except Exception as e:
    print(f"✗ Error saving CUT: {e}")

# Test loading the saved NPZ file
if npz_file.exists():
    try:
        loaded_pattern, loaded_metadata = load_pattern_npz(npz_file)
        print(f"✓ Successfully reloaded pattern from NPZ")
        print(f"  Loaded metadata: {loaded_metadata}")
        print_pattern_info(loaded_pattern, "Reloaded Pattern")
    except Exception as e:
        print(f"✗ Error reloading NPZ: {e}")

# ============================================================================
print_section_header("TUTORIAL COMPLETE")
# ============================================================================
print("""
Congratulations! You've completed the basic antenna pattern tutorial.

Key skills learned:
• Loading patterns from .CUT, .FFD, and .NPZ files
• Basic pattern visualization (gain, phase, axial ratio)
• Simple pattern analysis (peak gain, beamwidth, front-to-back ratio)
• Comparing patterns at different frequencies and phi cuts
• Converting between polarization types
• Saving patterns in multiple formats

Next steps:
• Try the polarization conversion tutorial for more advanced polarization handling
• Explore the synthetic pattern generation tutorial to create test patterns
• Learn about phase center optimization for precision applications

For questions or issues, refer to the AntennaPattern documentation or examples.
""")

print("="*60)
print("TUTORIAL SUMMARY")
print("="*60)
if 'main_pattern' in locals():
    print(f"Pattern analyzed: {main_pattern.polarization} at {analysis_freq/1e6:.1f} MHz")
    print(f"Peak gain: {np.max(main_pattern.get_gain_db('e_co').values):.2f} dBi")
    print(f"Files saved to: {output_dir}")
print("="*60)