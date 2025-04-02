"""
Example: Creating a synthetic antenna pattern using high-level parameters.

This example demonstrates how to use the create_synthetic_pattern function
to generate realistic antenna patterns from high-level parameters like:
- Peak gain
- Beamwidth
- Axial ratio
- Front-to-back ratio
- Sidelobe levels

These parameters are commonly used in antenna specifications without
requiring detailed electromagnetic modeling.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path if not already installed
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from antenna_pattern import AntennaPattern, create_synthetic_pattern

def demonstrate_basic_pattern():
    """Create and visualize a basic synthetic pattern."""
    # Define basic pattern parameters
    frequencies = np.array([8e9, 10e9, 12e9])  # X-band frequencies
    theta = np.linspace(-90, 90, 181)  # 1-degree step size
    phi = np.array([0, 45, 90, 135])  # Principal plane cuts
    
    # Create a synthetic pattern with default parameters
    e_theta, e_phi = create_synthetic_pattern(
        frequencies=frequencies,
        theta_angles=theta,
        phi_angles=phi,
        peak_gain_dbi=15.0,
        polarization='rhcp',
        beamwidth_deg=30.0,
        axial_ratio_db=1.0,
        front_to_back_db=25.0,
        sidelobe_level_db=-20.0
    )
    
    # Create the AntennaPattern object
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequencies,
        e_theta=e_theta,
        e_phi=e_phi
    )
    
    print(f"Created basic pattern with {pattern.polarization} polarization")
    
    # Plot the pattern
    plot_pattern(pattern, title="Basic Synthetic Pattern")
    
    return pattern

def demonstrate_variable_axial_ratio():
    """Create a pattern with variable axial ratio based on angle."""
    # Define pattern parameters
    frequencies = np.array([10e9])  # Single frequency for simplicity
    theta = np.linspace(-90, 90, 181)
    phi = np.array([0, 45, 90, 135])
    
    # Define axial ratio as a function of theta angle
    # 0.5 dB at boresight, 1 dB at 30 degrees, 3 dB at 90 degrees
    def ar_function(theta_deg):
        abs_theta = abs(theta_deg)
        if abs_theta < 30:
            # Linear interpolation from 0.5 to 1 dB
            return 0.5 + (abs_theta / 30) * 0.5
        else:
            # Linear interpolation from 1 to 3 dB
            return 1.0 + min(2.0, (abs_theta - 30) / 60 * 2.0)
    
    # Create the synthetic pattern
    e_theta, e_phi = create_synthetic_pattern(
        frequencies=frequencies,
        theta_angles=theta,
        phi_angles=phi,
        peak_gain_dbi=15.0,
        polarization='rhcp',
        beamwidth_deg=30.0,
        axial_ratio_db=ar_function,
        front_to_back_db=25.0,
        sidelobe_level_db=-25.0
    )
    
    # Create the AntennaPattern object
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequencies,
        e_theta=e_theta,
        e_phi=e_phi
    )
    
    print(f"Created pattern with variable axial ratio")
    
    # Plot the pattern and axial ratio
    plot_pattern(pattern, title="Pattern with Variable Axial Ratio")
    
    # Plot axial ratio vs angle
    plt.figure(figsize=(10, 6))
    test_angles = np.linspace(-90, 90, 181)
    ar_values = [ar_function(angle) for angle in test_angles]
    plt.plot(test_angles, ar_values)
    plt.title("Axial Ratio vs. Theta Angle")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Axial Ratio (dB)")
    plt.grid(True)
    plt.ylim(0, 3.5)
    plt.tight_layout()
    plt.show()
    
    return pattern

def demonstrate_frequency_scaling():
    """Create a pattern with frequency-dependent parameters."""
    # Define pattern parameters
    frequencies = np.array([8e9, 10e9, 12e9])
    theta = np.linspace(-90, 90, 181)
    phi = np.array([0, 45, 90, 135])
    
    # Define frequency-dependent parameters
    # Gain increases with frequency (approximately as frequency squared)
    peak_gain = np.array([12.0, 15.0, 17.5])  # dBi
    
    # Beamwidth decreases with frequency (approximately as 1/frequency)
    beamwidth = np.array([35.0, 28.0, 23.0])  # degrees
    
    # Front-to-back ratio increases with frequency
    front_to_back = np.array([20.0, 25.0, 30.0])  # dB
    
    # Create the synthetic pattern
    e_theta, e_phi = create_synthetic_pattern(
        frequencies=frequencies,
        theta_angles=theta,
        phi_angles=phi,
        peak_gain_dbi=peak_gain,
        polarization='rhcp',
        beamwidth_deg=beamwidth,
        axial_ratio_db=1.0,
        front_to_back_db=front_to_back,
        sidelobe_level_db=-20.0
    )
    
    # Create the AntennaPattern object
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequencies,
        e_theta=e_theta,
        e_phi=e_phi
    )
    
    print(f"Created pattern with frequency-dependent parameters")
    
    # Plot the pattern for all frequencies
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phi_idx = 0  # Use phi=0 cut
    for freq_idx, freq in enumerate(frequencies):
        ax.plot(theta, 
                pattern.get_gain_db('e_co').values[freq_idx, :, phi_idx],
                label=f"{freq/1e9:.1f} GHz (BW={beamwidth[freq_idx]:.1f}°)")
    
    ax.set_title("Frequency-Dependent Pattern Parameters")
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Gain (dBi)")
    ax.set_ylim(-40, 25)  # Increased upper limit to show full peak gain
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return pattern

def plot_pattern(pattern, title="Synthetic Antenna Pattern"):
    """Plot a 2x2 grid of pattern cuts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Use middle frequency if multiple frequencies
    freq_idx = len(pattern.frequencies) // 2
    
    # Plot principal plane cuts
    phi_cuts = [0, 90]
    phi_labels = ["E-plane", "H-plane"]
    
    for i, (phi_cut, phi_label) in enumerate(zip(phi_cuts, phi_labels)):
        p_idx = np.where(pattern.phi_angles == phi_cut)[0][0]
        
        # Plot co-pol with solid line
        axes[0, i].plot(pattern.theta_angles, 
                       pattern.get_gain_db('e_co').values[freq_idx, :, p_idx],
                       label="Co-pol", linewidth=2)
        
        # Plot cross-pol with dotted line
        axes[0, i].plot(pattern.theta_angles, 
                       pattern.get_gain_db('e_cx').values[freq_idx, :, p_idx],
                       label="Cross-pol", linestyle=':', linewidth=2)
        
        axes[0, i].set_title(f"{phi_label} (Phi={phi_cut}°) at {pattern.frequencies[freq_idx]/1e9:.1f} GHz")
        axes[0, i].set_xlabel("Theta (degrees)")
        axes[0, i].set_ylabel("Gain (dBi)")
        axes[0, i].set_ylim(-40, 25)  # Increased upper limit to ensure main beam is fully visible
        axes[0, i].grid(True)
        axes[0, i].legend()
    
    # Plot diagonal cuts
    phi_cuts = [45, 135]
    phi_labels = ["45° cut", "135° cut"]
    
    for i, (phi_cut, phi_label) in enumerate(zip(phi_cuts, phi_labels)):
        p_idx = np.where(pattern.phi_angles == phi_cut)[0][0]
        
        # Plot co-pol with solid line
        axes[1, i].plot(pattern.theta_angles, 
                       pattern.get_gain_db('e_co').values[freq_idx, :, p_idx],
                       label="Co-pol", linewidth=2)
        
        # Plot cross-pol with dotted line
        axes[1, i].plot(pattern.theta_angles, 
                       pattern.get_gain_db('e_cx').values[freq_idx, :, p_idx],
                       label="Cross-pol", linestyle=':', linewidth=2)
        
        axes[1, i].set_title(f"{phi_label} at {pattern.frequencies[freq_idx]/1e9:.1f} GHz")
        axes[1, i].set_xlabel("Theta (degrees)")
        axes[1, i].set_ylabel("Gain (dBi)")
        axes[1, i].set_ylim(-40, 25)  # Increased upper limit to ensure main beam is fully visible
        axes[1, i].grid(True)
        axes[1, i].legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """Run all pattern creation examples."""
    print("Creating basic synthetic pattern...")
    basic_pattern = demonstrate_basic_pattern()
    
    print("\nCreating pattern with variable axial ratio...")
    var_ar_pattern = demonstrate_variable_axial_ratio()
    
    print("\nCreating pattern with frequency-dependent parameters...")
    freq_pattern = demonstrate_frequency_scaling()
    
    print("\nSynthetic pattern examples complete.")
    return basic_pattern

if __name__ == "__main__":
    main()