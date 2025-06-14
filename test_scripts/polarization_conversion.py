"""
Example: Converting between different polarization types.

This example demonstrates how to convert an antenna pattern
between different polarization representations:
- RHCP/LHCP (Right/Left Hand Circular Polarization)
- X/Y (Linear, Ludwig-3)
- Theta/Phi (Spherical)
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

# Import the function to create our test pattern
from create_pattern_util import create_horn_pattern

def compare_polarizations(pattern):
    """
    Convert between different polarization types and visualize.
    
    Args:
        pattern: Original AntennaPattern
    """
    # List of polarizations to convert to
    polarizations = ["rhcp", "lhcp", "x", "y", "theta", "phi"]
    
    # Convert the pattern to each polarization
    converted_patterns = {}
    for pol in polarizations:
        converted_patterns[pol] = pattern.change_polarization(pol)
    
    # Create visualization to compare
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), layout="constrained")
    axes = axes.flatten()
    
    # Select frequency and phi cut for visualization
    freq_idx = 1  # 10 GHz
    phi_idx = 0   # 0 degrees
    
    # Plot each polarization
    for i, pol in enumerate(polarizations):
        # Plot the co-polarized component
        axes[i].plot(pattern.theta_angles, 
                    converted_patterns[pol].get_gain_db('e_co').values[freq_idx, :, phi_idx],
                    label="Co-polarized")
        
        # Plot the cross-polarized component
        axes[i].plot(pattern.theta_angles, 
                    converted_patterns[pol].get_gain_db('e_cx').values[freq_idx, :, phi_idx],
                    label="Cross-polarized", 
                    linestyle='--')
        
        axes[i].set_title(f"Polarization: {pol.upper()}")
        axes[i].set_xlabel("Theta (degrees)")
        axes[i].set_ylabel("Gain (dB)")
        axes[i].set_ylim(-40, 25)  # Increased upper limit to show full peak gain
        axes[i].grid(True)
        axes[i].legend()
    
    plt.suptitle(f"Polarization Comparison at {pattern.frequencies[freq_idx]/1e9:.1f} GHz, Phi={pattern.phi_angles[phi_idx]}°")
    plt.show()
    
    return converted_patterns

def main():
    """Demonstrate polarization conversion."""
    # Create the horn pattern (initially RHCP polarization)
    pattern = create_horn_pattern()
    print(f"Created horn pattern with {pattern.polarization} polarization")
    
    # Convert between polarizations and visualize
    converted_patterns = compare_polarizations(pattern)
    
    print("\nPolarization conversions complete!")
    print(f"Original polarization: {pattern.polarization}")
    for pol, converted in converted_patterns.items():
        print(f"Converted to {pol} polarization")
    
    plt.show()

if __name__ == "__main__":
    main()