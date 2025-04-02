"""
Example: Scaling the gain values of an antenna pattern.

This example demonstrates four ways to scale antenna pattern gain:
1. Uniform scaling (same value applied everywhere)
2. Frequency-dependent scaling
3. Frequency-dependent scaling with custom frequency points (interpolated)
4. 2D scaling with both frequency and phi dependency
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
from create_pattern_util import create_horn_pattern

def uniform_scaling(pattern, scale_value):
    """
    Apply uniform scaling to the pattern.
    
    Args:
        pattern: Original AntennaPattern
        scale_value: Scaling value in dB
        
    Returns:
        AntennaPattern: Scaled pattern
    """
    scaled_pattern = pattern.scale_pattern(scale_value)
    print(f"Applied uniform scaling of {scale_value} dB")
    
    return scaled_pattern

def frequency_dependent_scaling(pattern):
    """
    Apply frequency-dependent scaling to the pattern.
    
    Args:
        pattern: Original AntennaPattern
        
    Returns:
        AntennaPattern: Scaled pattern
    """
    # Create frequency-dependent scaling values
    # Increase gain more at higher frequencies
    scale_by_freq = np.array([2.0, 4.0, 8.0])  # Different value for each frequency
    
    scaled_pattern = pattern.scale_pattern(scale_by_freq)
    print("Applied frequency-dependent scaling:", scale_by_freq, "dB")
    
    return scaled_pattern

def interpolated_freq_scaling(pattern):
    """
    Apply frequency-dependent scaling using custom frequency points.
    
    Args:
        pattern: Original AntennaPattern
        
    Returns:
        AntennaPattern: Scaled pattern
    """
    # Define custom frequency points and scaling values
    custom_freqs = np.array([7e9, 10e9, 13e9])  # Different from pattern frequencies
    custom_scaling = np.array([0.5, 2.0, 3.5])  # Values at those frequencies
    
    scaled_pattern = pattern.scale_pattern(custom_scaling, freq_scale=custom_freqs)
    print("Applied interpolated frequency scaling at custom points:")
    for f, s in zip(custom_freqs/1e9, custom_scaling):
        print(f"  {f:.1f} GHz: {s:.1f} dB")
    
    return scaled_pattern

def freq_phi_dependent_scaling(pattern):
    """
    Apply 2D scaling with both frequency and phi dependency.
    
    Args:
        pattern: Original AntennaPattern
        
    Returns:
        AntennaPattern: Scaled pattern
    """
    # Create a 2D array with different scaling for each freq/phi combination
    # Format: [frequency, phi]
    n_freq = len(pattern.frequencies)
    n_phi = len(pattern.phi_angles)
    
    # Create a matrix where:
    # - Scaling increases with frequency
    # - Scaling varies sinusoidally with phi
    scale_2d = np.zeros((n_freq, n_phi))
    
    for f_idx in range(n_freq):
        for p_idx, phi in enumerate(pattern.phi_angles):
            # Base scaling that increases with frequency (1, 2, 3 dB)
            base_scale = f_idx + 1.0
            # Add phi-dependent variation (±0.5 dB)
            phi_variation = 0.5 * np.sin(np.radians(phi))
            scale_2d[f_idx, p_idx] = base_scale + phi_variation
    
    scaled_pattern = pattern.scale_pattern(scale_2d, freq_scale=pattern.frequencies)
    print("Applied 2D scaling with frequency and phi dependency:")
    for f_idx, freq in enumerate(pattern.frequencies/1e9):
        print(f"  {freq:.1f} GHz:", [f"{val:.2f}" for val in scale_2d[f_idx]])
    
    return scaled_pattern

def visualize_scaling_methods(original, uniform, freq_dep, interp, two_d):
    """
    Visualize the effects of different scaling methods.
    
    Args:
        original: Original pattern
        uniform: Uniformly scaled pattern
        freq_dep: Frequency-dependent scaled pattern
        interp: Interpolated frequency scaled pattern
        two_d: 2D (freq+phi) scaled pattern
    """
    # Create figure with 5 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), layout="constrained")
    axes = axes.flatten()
    
    # Select phi cut for visualization
    phi_idx = 0  # 0 degrees
    
    # Plot original pattern for all frequencies
    for f_idx, freq in enumerate(original.frequencies/1e9):
        axes[0].plot(original.theta_angles,
                    original.get_gain_db('e_co').values[f_idx, :, phi_idx],
                    label=f"{freq:.1f} GHz")
    
    axes[0].set_title("Original Pattern")
    axes[0].set_xlabel("Theta (degrees)")
    axes[0].set_ylabel("Gain (dB)")
    axes[0].set_ylim(-40, 30)
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot uniformly scaled pattern
    for f_idx, freq in enumerate(uniform.frequencies/1e9):
        axes[1].plot(uniform.theta_angles,
                    uniform.get_gain_db('e_co').values[f_idx, :, phi_idx],
                    label=f"{freq:.1f} GHz")
    
    axes[1].set_title("Uniform Scaling (4 dB)")
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Gain (dB)")
    axes[1].set_ylim(-40, 30)
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot frequency-dependent scaled pattern
    for f_idx, freq in enumerate(freq_dep.frequencies/1e9):
        axes[2].plot(freq_dep.theta_angles,
                    freq_dep.get_gain_db('e_co').values[f_idx, :, phi_idx],
                    label=f"{freq:.1f} GHz")
    
    axes[2].set_title("Frequency-Dependent Scaling")
    axes[2].set_xlabel("Theta (degrees)")
    axes[2].set_ylabel("Gain (dB)")
    axes[2].set_ylim(-40, 30)
    axes[2].grid(True)
    axes[2].legend()
    
    # Plot interpolated frequency scaled pattern
    for f_idx, freq in enumerate(interp.frequencies/1e9):
        axes[3].plot(interp.theta_angles,
                    interp.get_gain_db('e_co').values[f_idx, :, phi_idx],
                    label=f"{freq:.1f} GHz")
    
    axes[3].set_title("Interpolated Frequency Scaling")
    axes[3].set_xlabel("Theta (degrees)")
    axes[3].set_ylabel("Gain (dB)")
    axes[3].set_ylim(-40, 30)
    axes[3].grid(True)
    axes[3].legend()
    
    # Plot 2D scaled pattern for a specific frequency
    freq_idx = 1  # 10 GHz
    
    for p_idx, phi in enumerate(two_d.phi_angles):
        axes[4].plot(two_d.theta_angles,
                    two_d.get_gain_db('e_co').values[freq_idx, :, p_idx],
                    label=f"Phi={phi}°")
    
    axes[4].set_title(f"2D Scaling at {two_d.frequencies[freq_idx]/1e9:.1f} GHz")
    axes[4].set_xlabel("Theta (degrees)")
    axes[4].set_ylabel("Gain (dB)")
    axes[4].set_ylim(-40, 30)
    axes[4].grid(True)
    axes[4].legend()
    
    # Hide the unused subplot
    axes[5].axis('off')
    
    plt.suptitle("Comparison of Pattern Scaling Methods")
    plt.show()


def main():
    """Demonstrate different methods of scaling antenna pattern gain."""
    # Create the horn pattern
    pattern = create_horn_pattern()
    print(f"Created horn pattern with {pattern.polarization} polarization")
    
    # Apply different scaling methods
    uniform_scaled = uniform_scaling(pattern, 4.0)
    freq_dep_scaled = frequency_dependent_scaling(pattern)
    interp_scaled = interpolated_freq_scaling(pattern)
    two_d_scaled = freq_phi_dependent_scaling(pattern)
    
    # Visualize the results
    visualize_scaling_methods(pattern, uniform_scaled, freq_dep_scaled, 
                             interp_scaled, two_d_scaled)
    
    print("\nGain scaling demonstration complete!")
    
    plt.show()

if __name__ == "__main__":
    main()