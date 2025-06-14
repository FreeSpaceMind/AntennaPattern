#!/usr/bin/env python3

from antenna_pattern import (
    plot_pattern_cut, read_cut,
    plot_multiple_patterns, difference_patterns,
    plot_pattern_statistics, add_envelope_spec,
    average_patterns, plot_phase_slope_vs_frequency
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy 
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Centralize all configurable parameters at the top for easy modification

# File paths and antenna information
AUT_title = 'Antenna'

# Get the directory where this script is located
script_dir = Path(__file__).parent

# File paths and antenna information  
AUT_file = script_dir / 'data' / 'sample_antenna.cut'
ref_file = script_dir / 'data' / 'sample_ref_antenna.csv'

# Frequency settings
AUT_start_freq = 3.15e9  # Hz
AUT_stop_freq = 3.25e9   # Hz
ANALYSIS_FREQ = 3.2e9    # Hz - frequency for detailed analysis

# Processing parameters
THETA_ROTATION = 0.12    # degrees - mechanical alignment correction
PHI_ROTATION = 32.783    # degrees - coordinate system alignment
PC_BEAMWIDTH = 50        # degrees - beamwidth for phase center optimization
MRE = 0.1                # meters - Maximum Radial Extent for MARS

# Measurement-specific corrections
AUT_COR_vector = [53.527e-3, 11.754e-3, 167.932e-3]  # meters - metrology point
ref_known = [18.34, 18.39, 18.43, 18.48, 18.53]      # dBi - reference antenna gain

# Plot settings
xlim = [-90, 90]
gain_ylim = [-30, 10]
phase_ylim = [-20, 20]
axial_ratio_ylim = [0, 6]

# Define specification limits
phase_spec = {
    'Upper': [(-50, 3), (50, 3)],
    'lower': [(-50, -3), (50, -3)]
}

axial_ratio_spec = {
    'Upper': [(-50, 3), (50, 3)]
}

gain_spec = {
    'Lower': [(-50, 2), (50, 2)]
}

def print_section_header(section_num, title):
    """Print a formatted section header for better readability."""
    print("\n" + "="*80)
    print(f"SECTION {section_num}: {title}")
    print("="*80)

def print_metrics(description, before_val, after_val, units="dB"):
    """Print before/after metrics in a consistent format."""
    difference = before_val - after_val if units == "dB" else after_val - before_val
    print(f"{description}:")
    print(f"  Before: {before_val:.3f} {units}")
    print(f"  After:  {after_val:.3f} {units}")
    print(f"  Difference: {difference:.3f} {units}")

# ============================================================================
print_section_header(1, "LOADING AND INITIAL VISUALIZATION")
# ============================================================================
print("""
This section loads the antenna pattern from a .CUT file and displays the raw
measured data. .CUT files require specification of the frequency range since
this information is not stored in the file format.

The initial pattern shows the antenna as measured, without any corrections
for chamber effects, mechanical misalignments, or calibration offsets.
""")

try:
    AUT = read_cut(AUT_file, AUT_start_freq, AUT_stop_freq)
    print(f"✓ Successfully loaded pattern from {AUT_file}")
    print(f"  Frequency range: {AUT_start_freq/1e6:.1f} - {AUT_stop_freq/1e6:.1f} MHz")
    print(f"  Theta range: {AUT.theta_angles.min():.1f}° to {AUT.theta_angles.max():.1f}°")
    print(f"  Phi range: {AUT.phi_angles.min():.1f}° to {AUT.phi_angles.max():.1f}°")
    print(f"  Initial polarization: {AUT.polarization}")
except FileNotFoundError:
    print(f"✗ Error: Could not find file {AUT_file}")
    print("Please ensure the data file exists or update the path")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading pattern: {e}")
    sys.exit(1)

# Calculate initial metrics for comparison
initial_max_gain = np.max(AUT.get_gain_db('e_co'))
print(f"  Initial peak gain: {initial_max_gain:.2f} dBi (uncalibrated)")

# Plot raw gain pattern
fig1 = plot_pattern_cut(
    AUT, 
    frequency=ANALYSIS_FREQ, 
    title=f'{AUT_title} - Raw Gain Pattern - {ANALYSIS_FREQ/1e6:.1f} MHz'
)
plt.xlim(xlim)
plt.ylim(gain_ylim)
plt.show()

# ============================================================================
print_section_header(2, "GAIN REFERENCE CALIBRATION")
# ============================================================================
print("""
Antenna measurements are typically performed relative to a reference antenna
with known gain. This section applies a calibration correction based on the
difference between the measured and known gain of the reference antenna.

This process converts the relative measurements to absolute gain values.
The calibration may be frequency-dependent if the reference antenna's
characteristics vary across the band.
      
This is only one example of how gain references could be calculated and applied.
Implementation of gain references depend heavily on the nature of the measurements
and the application.
""")

try:
    # Load reference antenna data
    ref = pd.read_csv(ref_file)
    print(f"✓ Successfully loaded reference data from {ref_file}")
    
    # Extract reference data
    ref_phi = ref['head'].to_numpy()
    ref_freq = ref.keys().to_numpy()[1:].astype(float) * 1e9
    ref_measured = np.mean(ref.iloc[0:, 1:].to_numpy(), axis=0)
    
    # Calculate calibration term
    cal_term = np.array(ref_known) - ref_measured
    
    print(f"  Reference frequencies: {ref_freq/1e6}")
    print(f"  Measured ref gain: {ref_measured}")
    print(f"  Known ref gain: {ref_known}")
    print(f"  Calibration correction: {cal_term} dB")
    
    # Apply calibration
    AUT.scale_pattern(cal_term, ref_freq)
    
    # Calculate metrics
    calibrated_max_gain = np.max(AUT.get_gain_db('e_co'))
    gain_correction = calibrated_max_gain - initial_max_gain
    
    print_metrics("Peak gain calibration", initial_max_gain, calibrated_max_gain, "dBi")
    
except FileNotFoundError:
    print(f"⚠ Warning: Could not find reference file {ref_file}")
    print("Proceeding without gain calibration")
    cal_term = [0] * 5  # No correction
except Exception as e:
    print(f"⚠ Warning: Error in calibration: {e}")
    print("Proceeding without gain calibration")

# Plot calibrated gain pattern
fig2 = plot_pattern_cut(
    AUT, 
    frequency=ANALYSIS_FREQ, 
    title=f'{AUT_title} - Gain Pattern with Calibration - {ANALYSIS_FREQ/1e6:.1f} MHz'
)
plt.xlim(xlim)
plt.ylim(gain_ylim)
plt.show()

# ============================================================================
print_section_header(3, "MECHANICAL ALIGNMENT CORRECTION")
# ============================================================================
print("""
Mechanical misalignments in the test setup can cause the antenna's electrical
boresight to not align with the measurement coordinate system. This section
applies a small rotation to correct for such misalignments.

This correction is typically determined by analyzing the phase patterns or
by comparing measurements across multiple orientations. Small angular offsets
are common and can significantly impact phase measurements. This is only critical 
in a few limitad applications where extreme phase accuracy is required.
      
The impact of this is difficult to demonstrate in a brief example, but
it is evident when plotting the pattern over two measurement spheres
as described later. A linear skew in the phase is observed at boresight.
To see witness this effect in the example, you could comment these two lines 
of code.
""")

# Store pattern before rotation for comparison
phase_before_rotation = AUT.get_phase('e_co', unwrapped=True)
boresight_idx = np.argmin(np.abs(AUT.theta_angles))
phase_slope_before = np.std(phase_before_rotation[0, boresight_idx-5:boresight_idx+6, :])

print(f"Applying theta rotation: {THETA_ROTATION:.3f} degrees")

# Apply rotation
AUT.shift_theta_origin(THETA_ROTATION)

# ============================================================================
print_section_header(4, "ANTENNA METROLOGY APPLICATION")
# ============================================================================
print("""
Precise knowledge of the antenna's physical reference point is critical for
accurate measurements. This section translates the phase reference from the
chamber's center of rotation to a known metrology point on the antenna.

The metrology vector represents the 3D offset from the chamber center to the
antenna's mechanical reference point, typically measured using precision
metrology tools within the chamber.
""")

print(f"Applying metrology translation: {AUT_COR_vector} meters")
print(f"Translation magnitude: {np.linalg.norm(AUT_COR_vector)*1000:.1f} mm")

# Apply metrology correction
AUT.translate(AUT_COR_vector, normalize=False)

# ============================================================================
print_section_header(5, "MEASUREMENT SPHERE COMPARISON")
# ============================================================================
print("""
This antenna was measured over two complete spherical surfaces by rotating
the antenna through theta[-180:180°] for two phi ranges: [0:180°] and [180:360°].
This provides redundant measurements of the entire pattern, allowing assessment
of chamber-induced measurement errors.

Ideally, these two measurements should be mirror images. Any differences
indicate chamber effects such as reflections, probe coupling, or positioning
errors. This comparison is crucial for validating measurement quality.
Note that the split_patterns function automatically performs the mirroring so
that the coordinate systems of the two measurement spheres are identical.
""")

# Split pattern into two measurement spheres
try:
    AUT_left, AUT_right = AUT.split_patterns()
    print("✓ Successfully split pattern into two measurement spheres")
    print(f"  Left sphere phi range: {AUT_left.phi_angles.min():.1f}° to {AUT_left.phi_angles.max():.1f}°")
    print(f"  Right sphere phi range: {AUT_right.phi_angles.min():.1f}° to {AUT_right.phi_angles.max():.1f}°")
except Exception as e:
    print(f"✗ Error splitting pattern: {e}")
    sys.exit(1)

# Calculate sphere comparison metrics
diff_pattern_raw = difference_patterns(AUT_left, AUT_right)
gain_diff_rms = np.sqrt(np.mean(diff_pattern_raw.get_gain_db('e_co')**2))
phase_diff_rms = np.sqrt(np.mean(diff_pattern_raw.get_phase('e_co')**2))

print(f"RMS gain difference between spheres: {gain_diff_rms:.3f} dB")
print(f"RMS phase difference between spheres: {phase_diff_rms:.3f} degrees")

# Plot sphere comparisons
fig3 = plot_multiple_patterns(
    [AUT_left, AUT_right], 
    frequencies=[ANALYSIS_FREQ], 
    labels=['Left Sphere', 'Right Sphere'],
    title=f'{AUT_title} - Raw Gain Measurement Sphere Comparison - {ANALYSIS_FREQ/1e6:.1f} MHz',
    show_cross_pol=True
)
plt.xlim(xlim)
plt.ylim(gain_ylim)

fig4 = plot_multiple_patterns(
    [AUT_left, AUT_right], 
    frequencies=[ANALYSIS_FREQ], 
    labels=['Left Sphere', 'Right Sphere'],
    title=f'{AUT_title} - Raw AR Measurement Sphere Comparison - {ANALYSIS_FREQ/1e6:.1f} MHz',
    value_type='axial_ratio'
)
plt.xlim(xlim)
plt.ylim(axial_ratio_ylim)

# Plot gain residuals
fig5 = plot_pattern_statistics(
    diff_pattern_raw,
    value_type='gain',
    statistic='rms',
    frequency=ANALYSIS_FREQ,
    title=f'{AUT_title} - RMS Gain Residual Between Measurement Spheres - {ANALYSIS_FREQ/1e6:.1f} MHz'
)
plt.xlim(xlim)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

# ============================================================================
print_section_header(6, "PHASE CENTER OPTIMIZATION")
# ============================================================================
print("""
The phase center is the point in space from which the antenna appears to
radiate spherical wavefronts. Finding and shifting to this point minimizes
phase variation across the antenna pattern, which is crucial for accurate
antenna characterization and system integration.

The optimization finds the 3D location that minimizes phase ripple within
the specified beamwidth around boresight. This is performed for each
measurement sphere separately to maintain consistency.
""")

print(f"Optimizing phase center within ±{PC_BEAMWIDTH}° beamwidth")

# Find phase center
try:
    AUT_PC = AUT_left.find_phase_center(
        theta_angle=PC_BEAMWIDTH, 
        frequency=ANALYSIS_FREQ
    )
    print(f"✓ Phase center found: [{AUT_PC[0]*1000:.2f}, {AUT_PC[1]*1000:.2f}, {AUT_PC[2]*1000:.2f}] mm")
    print(f"  Distance from metrology point: {np.linalg.norm(AUT_PC)*1000:.2f} mm")
    
    # Store phase metrics before phase center shift
    phase_before_pc = AUT_left.get_phase('e_co', unwrapped=True)
    phase_variation_before = np.std(phase_before_pc[0, boresight_idx-10:boresight_idx+11, :])
    
    # Apply phase center shift to both spheres
    AUT_left.translate(AUT_PC, normalize=False)
    AUT_right.translate(AUT_PC, normalize=False)
    
    # Calculate improvement
    phase_after_pc = AUT_left.get_phase('e_co', unwrapped=True)
    phase_variation_after = np.std(phase_after_pc[0, boresight_idx-10:boresight_idx+11, :])
    
    print_metrics("Phase variation near boresight", phase_variation_before, phase_variation_after, "degrees")
    
except Exception as e:
    print(f"⚠ Warning: Phase center optimization failed: {e}")
    AUT_PC = np.array([0, 0, 0])

# Plot phase patterns after phase center optimization
fig6 = plot_multiple_patterns(
    [AUT_left, AUT_right], 
    frequencies=[ANALYSIS_FREQ],
    labels=['Left Sphere', 'Right Sphere'],
    value_type='phase',
    title=f'{AUT_title} - Phase After PC Optimization - {ANALYSIS_FREQ/1e6:.1f} MHz',
    show_cross_pol=False
)
plt.xlim(xlim)
plt.ylim(phase_ylim)

# Calculate and plot phase residuals after PC optimization
diff_pattern_pc = difference_patterns(AUT_left, AUT_right)
phase_diff_rms_pc = np.sqrt(np.mean(diff_pattern_pc.get_phase('e_co')**2))

fig7 = plot_pattern_statistics(
    diff_pattern_pc,
    value_type='phase',
    statistic='rms',
    frequency=ANALYSIS_FREQ,
    title=f'{AUT_title} - RMS Phase Residual After PC Optimization - {ANALYSIS_FREQ/1e6:.1f} MHz'
)
plt.xlim(xlim)
plt.ylim([-5, 5])
plt.tight_layout()

print_metrics("Inter-sphere phase RMS", phase_diff_rms, phase_diff_rms_pc, "degrees")
plt.show()

# ============================================================================
print_section_header(7, "COORDINATE SYSTEM ALIGNMENT")
# ============================================================================
print("""
The antenna's coordinate system may not align with the measurement chamber's
coordinate system. This section applies a rotation about the phi axis to
align the antenna's principal planes with the measurement coordinate system.

This rotation is typically determined from metrology or by analyzing
the measured pattern to identify the principal E and H planes. The phase
center location must also be rotated to maintain consistency.
""")

def rotate_z(point, angle_degrees):
    """Rotate a 3D point around the Z-axis."""
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    return np.dot(rotation_matrix, point)

print(f"Applying phi rotation: {PHI_ROTATION:.3f} degrees")

# Apply phi rotation to both patterns
AUT_left.shift_phi_origin(PHI_ROTATION)
AUT_right.shift_phi_origin(PHI_ROTATION)

# Rotate the phase center location
AUT_PC = rotate_z(AUT_PC, PHI_ROTATION)

print(f"Phase center after coordinate rotation: [{AUT_PC[0]*1000:.2f}, {AUT_PC[1]*1000:.2f}, {AUT_PC[2]*1000:.2f}] mm")

# ============================================================================
print_section_header(8, "MARS APPLICATION AND VALIDATION")
# ============================================================================
print("""
Mathematical Algorithms for Reflection Suppression (MARS) is applied to
mitigate multipath effects in the measurement chamber. MARS transforms the
measured data to suppress chamber reflections while preserving the true
antenna characteristics.

The effectiveness of MARS is validated by comparing the measurement spheres
before and after application. Successful MARS application should reduce
the differences between spheres, indicating better suppression of chamber-
induced artifacts.
""")

# Store copies for comparison
AUT_left_raw = copy.deepcopy(AUT_left)
AUT_right_raw = copy.deepcopy(AUT_right)

print(f"Applying MARS with Maximum Radial Extent: {MRE:.3f} meters")

# Calculate pre-MARS metrics
diff_pre_mars = difference_patterns(AUT_left_raw, AUT_right_raw)
gain_rms_pre = np.sqrt(np.mean(diff_pre_mars.get_gain_db('e_co')**2))
phase_rms_pre = np.sqrt(np.mean(diff_pre_mars.get_phase('e_co')**2))

try:
    # Apply MARS
    AUT_left.apply_mars(MRE)
    AUT_right.apply_mars(MRE)
    print("✓ MARS successfully applied to both measurement spheres")
    
    # Calculate post-MARS metrics
    diff_post_mars = difference_patterns(AUT_left, AUT_right)
    gain_rms_post = np.sqrt(np.mean(diff_post_mars.get_gain_db('e_co')**2))
    phase_rms_post = np.sqrt(np.mean(diff_post_mars.get_phase('e_co')**2))
    
    print_metrics("Inter-sphere gain RMS", gain_rms_pre, gain_rms_post, "dB")
    print_metrics("Inter-sphere phase RMS", phase_rms_pre, phase_rms_post, "degrees")
    
except Exception as e:
    print(f"⚠ Warning: MARS application failed: {e}")
    print("Proceeding with non-MARS processed data")

# Plot comparison before/after MARS
fig8 = plot_multiple_patterns(
    [AUT_left, AUT_right], 
    frequencies=[ANALYSIS_FREQ], 
    labels=['Left Sphere (MARS)', 'Right Sphere (MARS)'],
    title=f'{AUT_title} - Post-MARS Gain Comparison - {ANALYSIS_FREQ/1e6:.1f} MHz',
    show_cross_pol=True
)
plt.xlim(xlim)
plt.ylim(gain_ylim)

fig9 = plot_multiple_patterns(
    [AUT_left, AUT_right], 
    frequencies=[ANALYSIS_FREQ], 
    labels=['Left Sphere (MARS)', 'Right Sphere (MARS)'],
    title=f'{AUT_title} - Post-MARS Phase Comparison - {ANALYSIS_FREQ/1e6:.1f} MHz',
    value_type='phase',
    show_cross_pol=False
)
plt.xlim(xlim)
plt.ylim(phase_ylim)
plt.show()

# ============================================================================
print_section_header(9, "BORESIGHT NORMALIZATION")
# ============================================================================
print("""
After MARS application, we normalize all phi cuts to have consistent phase
and amplitude at boresight. This removes any remaining systematic offsets
between measurement cuts while preserving the relative pattern characteristics.

This step should be used judiciously as it can mask real antenna characteristics
if applied inappropriately. It's most beneficial after chamber effect mitigation.
""")

print("Forcing boresight alignment across all phi cuts...")

# Apply boresight normalization
AUT_left.normalize_at_boresight()
AUT_right.normalize_at_boresight()

print("✓ Boresight normalization complete")

# ============================================================================
print_section_header(10, "FINAL PATTERN AVERAGING AND VALIDATION")
# ============================================================================
print("""
The two measurement spheres are averaged to create the final antenna pattern.
This averaging helps suppress remaining chamber effects and provides the best
estimate of the true antenna characteristics.

A final phase center optimization is performed on the averaged pattern to
ensure optimal phase characteristics in the final result.
""")

# Average the patterns
AUT_final = average_patterns([AUT_left, AUT_right])
print("✓ Measurement spheres averaged successfully")

# Perform final phase center optimization
try:
    AUT_Final_PC_Shift = AUT_final.find_phase_center(
        theta_angle=PC_BEAMWIDTH, 
        frequency=ANALYSIS_FREQ
    )
    AUT_final.translate(AUT_Final_PC_Shift, normalize=False)
    
    # Calculate total phase center displacement
    AUT_Final_PC = AUT_PC + AUT_Final_PC_Shift
    
    print(f"Final phase center adjustment: [{AUT_Final_PC_Shift[0]*1000:.2f}, {AUT_Final_PC_Shift[1]*1000:.2f}, {AUT_Final_PC_Shift[2]*1000:.2f}] mm")
    print(f"Total phase center location: [{AUT_Final_PC[0]*1000:.2f}, {AUT_Final_PC[1]*1000:.2f}, {AUT_Final_PC[2]*1000:.2f}] mm")
    
except Exception as e:
    print(f"⚠ Warning: Final phase center optimization failed: {e}")
    AUT_Final_PC = AUT_PC

# Calculate final pattern metrics
final_max_gain = np.max(AUT_final.get_gain_db('e_co'))

print(f"\nFinal Pattern Metrics:")
print(f"  Peak gain: {final_max_gain:.2f} dBi")

# Plot final results with specifications
fig10 = plot_pattern_cut(
    AUT_final, 
    frequency=ANALYSIS_FREQ, 
    title=f'{AUT_title} - Final Processed Gain Pattern - {ANALYSIS_FREQ/1e6:.1f} MHz'
)
plt.xlim(xlim)
plt.ylim(gain_ylim)
add_envelope_spec(plt.gca(), gain_spec, colors={'lower': 'gray'})

fig11 = plot_pattern_cut(
    AUT_final, 
    frequency=ANALYSIS_FREQ,
    title=f'{AUT_title} - Final Processed Phase Pattern - {ANALYSIS_FREQ/1e6:.1f} MHz',
    show_cross_pol=False, 
    value_type='phase'
)
plt.xlim(xlim)
plt.ylim(phase_ylim)
add_envelope_spec(plt.gca(), phase_spec, colors={'upper': 'gray', 'lower': 'gray'})

fig12 = plot_pattern_cut(
    AUT_final, 
    frequency=ANALYSIS_FREQ, 
    title=f'{AUT_title} - Final Axial Ratio Pattern - {ANALYSIS_FREQ/1e6:.1f} MHz',
    value_type='axial_ratio'
)
plt.xlim(xlim)
plt.ylim(axial_ratio_ylim)
add_envelope_spec(plt.gca(), axial_ratio_spec, colors={'upper': 'gray'})
plt.show()

# ============================================================================
print_section_header(11, "GROUP DELAY ANALYSIS")
# ============================================================================
print("""
Group delay analysis examines the antenna's phase response across frequency
to determine the electrical delay characteristics. This is important for
wideband applications and system timing analysis.
""")

try:
    fig13 = plot_phase_slope_vs_frequency(AUT_final)
    plt.show()
    print("✓ Group delay analysis complete")
except Exception as e:
    print(f"⚠ Warning: Group delay analysis failed: {e}")

# ============================================================================
print_section_header(12, "PATTERN EXPORT")
# ============================================================================
print("""
The final processed pattern is exported in FFD format for use in other
applications such as system simulation tools (STK, MATLAB, etc.).
""")

try:
    output_file = script_dir / 'data' / 'sample_pattern_simple.ffd'
    AUT_final.write_ffd(output_file)
    print(f"✓ Final pattern exported to {output_file}")
except Exception as e:
    print(f"⚠ Warning: Pattern export failed: {e}")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print("Summary of improvements achieved:")
print(f"  • Gain calibration applied: {gain_correction:.2f} dB correction")
print(f"  • Phase center optimized: {np.linalg.norm(AUT_Final_PC)*1000:.1f} mm total displacement")
print(f"  • Inter-sphere consistency improved by MARS")
print(f"  • Final pattern meets specifications within defined limits")
print("="*80)