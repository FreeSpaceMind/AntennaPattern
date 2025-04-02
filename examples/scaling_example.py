"""
Example demonstrating how to scale an antenna pattern amplitude using different methods.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path if not already installed
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from antenna_pattern import AntennaPattern

# Create a simple antenna pattern
def create_test_pattern():
    """Create a simple cosine antenna pattern for demonstration."""
    # Create pattern parameters
    theta = np.linspace(-90, 90, 181)  # 1 degree steps
    phi = np.array([0, 45, 90])  # Three phi cuts
    frequency = np.array([2e9, 4e9, 6e9])  # Three frequencies
    
    # Initialize field components
    e_theta = np.zeros((len(frequency), len(theta), len(phi)), dtype=complex)
    e_phi = np.zeros((len(frequency), len(theta), len(phi)), dtype=complex)
    
    # Fill with a simple cosine pattern
    for f_idx, f_val in enumerate(frequency):
        for t_idx, t_val in enumerate(theta):
            for p_idx, p_val in enumerate(phi):
                # Simple cosine pattern with frequency-dependent amplitude
                e_theta[f_idx, t_idx, p_idx] = (1 + 0.2*f_idx) * np.cos(np.radians(t_val))
                # Very small cross-polarization
                e_phi[f_idx, t_idx, p_idx] = 0.05 * np.sin(np.radians(t_val))
    
    # Create antenna pattern
    return AntennaPattern(
        theta=theta,
        phi=phi,
        frequency=frequency,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization="theta"
    )

def main():
    """Demonstrate antenna pattern scaling."""
    # Create a test pattern
    pattern = create_test_pattern()
    
    # Plot the original pattern
    freq_idx = 0  # First frequency
    phi_idx = 0   # First phi cut
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Antenna Pattern Scaling Examples", fontsize=16)
    
    # Plot original pattern
    axes[0, 0].plot(pattern.theta_angles, 
                   pattern.get_gain_db('e_theta').values[freq_idx, :, phi_idx])
    axes[0, 0].set_title("Original Pattern")
    axes[0, 0].set_xlabel("Theta (degrees)")
    axes[0, 0].set_ylabel("Gain (dB)")
    axes[0, 0].grid(True)
    
    # Example 1: Scale by a single scalar value (3 dB)
    scaled_pattern1 = pattern.scale_pattern(3.0)
    axes[0, 1].plot(pattern.theta_angles,
                   scaled_pattern1.get_gain_db('e_theta').values[freq_idx, :, phi_idx])
    axes[0, 1].set_title("Scaled by 3 dB")
    axes[0, 1].set_xlabel("Theta (degrees)")
    axes[0, 1].set_ylabel("Gain (dB)")
    axes[0, 1].grid(True)
    
    # Example 2: Scale by frequency-dependent values
    # 2 dB at 2 GHz, 3 dB at 4 GHz, 4 dB at 6 GHz
    scale_by_freq = np.array([2.0, 3.0, 4.0])
    scaled_pattern2 = pattern.scale_pattern(scale_by_freq)
    
    # Plot all frequencies to show different scaling
    for f_idx, freq in enumerate(pattern.frequencies/1e9):
        axes[1, 0].plot(pattern.theta_angles,
                       scaled_pattern2.get_gain_db('e_theta').values[f_idx, :, phi_idx],
                       label=f"{freq} GHz")
    
    axes[1, 0].set_title("Frequency-Dependent Scaling")
    axes[1, 0].set_xlabel("Theta (degrees)")
    axes[1, 0].set_ylabel("Gain (dB)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Example 3: Scale with both frequency and phi dependency
    # Create a 2D array with different scaling for each freq/phi combination
    scale_2d = np.array([
        [1.0, 2.0, 3.0],  # For 2 GHz, phi=[0,45,90]
        [2.0, 3.0, 4.0],  # For 4 GHz, phi=[0,45,90]
        [3.0, 4.0, 5.0]   # For 6 GHz, phi=[0,45,90]
    ])
    
    scaled_pattern3 = pattern.scale_pattern(scale_2d, freq_scale=pattern.frequencies)
    
    # Plot phi cuts at middle frequency
    f_idx = 1  # 4 GHz
    for p_idx, phi in enumerate(pattern.phi_angles):
        axes[1, 1].plot(pattern.theta_angles,
                       scaled_pattern3.get_gain_db('e_theta').values[f_idx, :, p_idx],
                       label=f"Phi={phi}°")
    
    axes[1, 1].set_title("Frequency and Phi Dependent Scaling (4 GHz)")
    axes[1, 1].set_xlabel("Theta (degrees)")
    axes[1, 1].set_ylabel("Gain (dB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()