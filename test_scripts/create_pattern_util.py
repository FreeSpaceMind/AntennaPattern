"""
Utility module for creating and visualizing antenna patterns for the examples.
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

def create_horn_pattern():
    """
    Create a synthetic horn antenna pattern with 1 dB axial ratio.
    
    This function uses the create_synthetic_pattern utility to generate a 
    realistic horn antenna pattern with proper sidelobes and polarization.
    
    Returns:
        AntennaPattern: Synthetic horn antenna pattern
    """
    # Define pattern parameters
    theta = np.linspace(-90, 90, 361)  # 0.5 degree steps for better sidelobe resolution
    phi = np.array([0, 45, 90, 135])   # Principal plane cuts
    frequency = np.array([8e9, 10e9, 12e9])  # X-band frequencies
    
    # Define variable axial ratio as a function of theta angle
    # 0.5 dB at boresight, 1 dB at half-power beamwidth, 3 dB at far angles
    def ar_function(theta_deg):
        abs_theta = abs(theta_deg)
        if abs_theta < 15:  # Within half-power beamwidth
            # Linear interpolation from 0.5 to 1 dB
            return 0.5 + (abs_theta / 15) * 0.5
        else:
            # Linear interpolation from 1 to 3 dB
            return 1.0 + min(2.0, (abs_theta - 15) / 75 * 2.0)
    
    # Create the synthetic pattern
    e_theta, e_phi = create_synthetic_pattern(
        frequencies=frequency,
        theta_angles=theta,
        phi_angles=phi,
        peak_gain_dbi=np.array([12.0, 15.0, 17.5]),  # Frequency-dependent gain 
        polarization='rhcp',
        beamwidth_deg=np.array([35.0, 30.0, 25.0]),  # Frequency-dependent beamwidth
        axial_ratio_db=ar_function,
        front_to_back_db=25.0,
        sidelobe_level_db=-20.0
    )
    
    # Create the AntennaPattern object
    pattern = AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi
    )
    
    return pattern

def visualize_pattern(pattern, title="Synthetic Horn Antenna Pattern"):
    """
    Create a visualization of the pattern with co-pol and cross-pol on the same axes.
    
    Args:
        pattern: AntennaPattern to visualize
        title: Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot gain at middle frequency
    freq_idx = 1  # 10 GHz
    
    # Plot co-pol and cross-pol on the same axis for each phi cut
    phi_cuts = [0, 90]  # Show phi=0 and phi=90 cuts
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
        axes[0, i].set_ylim(-30, 25)  # Increased upper limit to ensure main beam is fully visible
        axes[0, i].grid(True)
        axes[0, i].legend()
    
    # Plot diagonal cuts
    phi_cuts = [45, 135]  # Show phi=45 and phi=135 cuts
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
        axes[1, i].set_ylim(-30, 25)  # Increased upper limit to ensure main beam is fully visible
        axes[1, i].grid(True)
        axes[1, i].legend()
    
    # Add a plot showing all frequencies at phi=0
    plt.figure(figsize=(10, 6))
    p_idx = 0  # Phi=0
    
    for f_idx, freq in enumerate(pattern.frequencies):
        plt.plot(pattern.theta_angles, 
                pattern.get_gain_db('e_co').values[f_idx, :, p_idx],
                label=f"{freq/1e9:.1f} GHz", linewidth=2)
    
    plt.title("Frequency Comparison (E-plane)")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Gain (dBi)")
    plt.ylim(-30, 25)
    plt.xlim(-60, 60)  # Focus on main beam and sidelobes
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()