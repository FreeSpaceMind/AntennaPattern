"""
Simple visualization of antenna pattern coordinate transformations using the
create_synthetic_pattern function from the package.
"""
import numpy as np
import matplotlib.pyplot as plt
from antenna_pattern import AntennaPattern, create_synthetic_pattern

# -------------------------------------------------
# Case 1: Central to Sided transformation
# -------------------------------------------------

# Create a pattern in central format (theta: -180:180, phi: 0:180)
theta_central = np.linspace(-180, 180, 181)  # 5-degree steps
phi_central = np.array([0, 45, 90, 135])  # Just a few phi cuts
frequency = np.array([10e9])  # 10 GHz

# Create synthetic pattern with recognizable features
e_theta_central, e_phi_central = create_synthetic_pattern(
    frequencies=frequency,
    theta_angles=theta_central,
    phi_angles=phi_central,
    peak_gain_dbi=15.0,           # Peak gain in dBi
    polarization='rhcp',          # Circular polarization
    beamwidth_deg=30.0,           # 3dB beamwidth
    axial_ratio_db=1.0,           # Axial ratio in dB
    front_to_back_db=25.0,        # Front-to-back ratio
    sidelobe_level_db=-20.0       # Sidelobe level
)

# Create AntennaPattern object
pattern_central = AntennaPattern(
    theta=theta_central,
    phi=phi_central,
    frequency=frequency,
    e_theta=e_theta_central,
    e_phi=e_phi_central,
    polarization="rhcp"
)

# Transform to sided format
pattern_sided = pattern_central.transform_coordinates('sided')

# -------------------------------------------------
# Case 2: Sided to Central transformation
# -------------------------------------------------

# Create a pattern in sided format (theta: 0:180, phi: 0:360)
theta_sided = np.linspace(0, 180, 181)  # Only positive theta
phi_sided = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # Full 360 degrees

# Create synthetic pattern with recognizable features
e_theta_sided, e_phi_sided = create_synthetic_pattern(
    frequencies=frequency,
    theta_angles=theta_sided,
    phi_angles=phi_sided,
    peak_gain_dbi=15.0,           # Peak gain in dBi
    polarization='rhcp',          # Circular polarization
    beamwidth_deg=30.0,           # 3dB beamwidth
    axial_ratio_db=1.0,           # Axial ratio in dB
    front_to_back_db=25.0,        # Front-to-back ratio
    sidelobe_level_db=-20.0       # Sidelobe level
)

# Create AntennaPattern object
pattern_sided_orig = AntennaPattern(
    theta=theta_sided,
    phi=phi_sided,
    frequency=frequency,
    e_theta=e_theta_sided,
    e_phi=e_phi_sided,
    polarization="rhcp"
)

# Transform to central format
pattern_central_new = pattern_sided_orig.transform_coordinates('central')

# -------------------------------------------------
# Visualization
# -------------------------------------------------

# Function to plot pattern cuts
def plot_pattern_cuts(pattern, ax, title):
    """Plot pattern cuts at different phi angles."""
    # Get gain in dB for co-pol component
    gain_db = pattern.get_gain_db('e_co').values[0]  # First frequency
    
    # Plot for each phi cut
    for p_idx, p_val in enumerate(pattern.phi_angles):
        ax.plot(pattern.theta_angles, gain_db[:, p_idx], label=f'φ={p_val:.0f}°')
    
    ax.set_title(title)
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Gain (dB)')
    ax.grid(True)
    ax.legend()

# Case 1: Plot central → sided transformation
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_pattern_cuts(pattern_central, ax1, 'Original Central Format\n(θ: -180° to 180°, φ: 0° to 180°)')
plot_pattern_cuts(pattern_sided, ax2, 'Transformed to Sided Format\n(θ: 0° to 180°, φ: 0° to 360°)')

plt.suptitle('Case 1: Central to Sided Transformation', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Case 2: Plot sided → central transformation
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

plot_pattern_cuts(pattern_sided_orig, ax3, 'Original Sided Format\n(θ: 0° to 180°, φ: 0° to 360°)')
plot_pattern_cuts(pattern_central_new, ax4, 'Transformed to Central Format\n(θ: -180° to 180°, φ: 0° to 180°)')

plt.suptitle('Case 2: Sided to Central Transformation', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.show()