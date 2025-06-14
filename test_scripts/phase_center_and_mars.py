"""
Example: Working with phase center and applying MARS algorithm.

This example demonstrates:
1. Introducing phase shifts to an antenna pattern
2. Finding the phase center
3. Shifting to the phase center
4. Applying the MARS algorithm to suppress chamber reflections

# Note that this is not an insightful example because this is not measured data.
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

def apply_phase_shift(pattern):
    """
    Introduce a phase shift to simulate an offset phase center.
    
    Args:
        pattern: Original AntennaPattern
        
    Returns:
        AntennaPattern: Pattern with phase shift
    """
    # Introduce a translation to shift the phase center
    # This simulates a horn where the phase center is not at the aperture
    translation = np.array([0.0, 0.0, 0.05])  # 5cm offset in z-direction
    
    shifted_pattern = pattern.translate(translation)
    print(f"Applied phase shift using translation: {translation} meters")
    
    return shifted_pattern

def visualize_phase(original_pattern, shifted_pattern):
    """
    Visualize the phase of original and shifted patterns.
    
    Args:
        original_pattern: Original antenna pattern
        shifted_pattern: Pattern with phase shift
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Select frequency and phi for visualization
    freq_idx = 1  # 10 GHz
    phi_idx = 0   # 0 degrees
    
    # Plot original phase
    original_phase = original_pattern.get_phase('e_co', unwrapped=True)
    axes[0].plot(original_pattern.theta_angles, 
                np.rad2deg(original_phase[freq_idx, :, phi_idx]))
    axes[0].set_title("Original Pattern Phase")
    axes[0].set_xlabel("Theta (degrees)")
    axes[0].set_ylabel("Phase (degrees)")
    axes[0].grid(True)
    
    # Plot shifted phase
    shifted_phase = shifted_pattern.get_phase('e_co', unwrapped=True)
    axes[1].plot(shifted_pattern.theta_angles, 
                np.rad2deg(shifted_phase[freq_idx, :, phi_idx]))
    axes[1].set_title("Shifted Pattern Phase")
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def find_and_apply_phase_center(shifted_pattern):
    """
    Find the phase center and shift the pattern to it.
    
    Args:
        shifted_pattern: Pattern with phase shift
        
    Returns:
        tuple: (pattern shifted to phase center, phase center coordinates)
    """
    # Find phase center for a 60-degree beamwidth
    theta_angle = 30.0  # +/- 30 degrees = 60 degrees beamwidth
    freq = 10e9  # 10 GHz
    
    # Find the phase center
    phase_center = shifted_pattern.find_phase_center(theta_angle, freq)
    print(f"Found phase center at: {phase_center} meters")
    
    # Shift pattern to the phase center
    corrected_pattern, translation = shifted_pattern.shift_to_phase_center(theta_angle, freq)
    print(f"Applied shift to phase center with translation: {translation} meters")
    
    return corrected_pattern, phase_center

def visualize_phase_center_correction(shifted_pattern, corrected_pattern):
    """
    Visualize the phase before and after phase center correction.
    
    Args:
        shifted_pattern: Pattern with phase shift
        corrected_pattern: Pattern shifted to phase center
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Select frequency and phi for visualization
    freq_idx = 1  # 10 GHz
    phi_idx = 0   # 0 degrees
    
    # Plot shifted pattern phase
    shifted_phase = shifted_pattern.get_phase('e_co', unwrapped=True)
    axes[0].plot(shifted_pattern.theta_angles, 
                np.rad2deg(shifted_phase[freq_idx, :, phi_idx]))
    axes[0].set_title("Before Phase Center Correction")
    axes[0].set_xlabel("Theta (degrees)")
    axes[0].set_ylabel("Phase (degrees)")
    axes[0].grid(True)
    
    # Plot corrected pattern phase
    corrected_phase = corrected_pattern.get_phase('e_co', unwrapped=True)
    axes[1].plot(corrected_pattern.theta_angles, 
                np.rad2deg(corrected_phase[freq_idx, :, phi_idx]))
    axes[1].set_title("After Phase Center Correction")
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def apply_mars_algorithm(pattern):
    """
    Apply the MARS algorithm to suppress chamber reflections.
    
    Args:
        pattern: Original pattern
        
    Returns:
        AntennaPattern: Pattern with MARS applied
    """
    # Estimate the maximum radial extent of the horn antenna
    # For a typical horn, this would be a few wavelengths
    wavelength = 3e8 / 10e9  # wavelength at 10 GHz = 3cm
    max_radial_extent = 3 * wavelength  # 3 wavelengths = 9cm
    
    # Apply MARS
    mars_pattern = pattern.apply_mars(max_radial_extent)
    print(f"Applied MARS algorithm with maximum radial extent: {max_radial_extent:.3f} meters")
    
    return mars_pattern

def visualize_mars_effect(original_pattern, mars_pattern):
    """
    Visualize the effect of MARS on the pattern.
    
    Args:
        original_pattern: Original pattern
        mars_pattern: Pattern with MARS applied
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Select frequency and phi for visualization
    freq_idx = 1  # 10 GHz
    phi_idx = 0   # 0 degrees
    
    # Plot original gain
    axes[0].plot(original_pattern.theta_angles, 
                original_pattern.get_gain_db('e_co').values[freq_idx, :, phi_idx],
                label="Original")
    axes[0].set_title("Gain Before MARS")
    axes[0].set_xlabel("Theta (degrees)")
    axes[0].set_ylabel("Gain (dB)")
    axes[0].set_ylim(-40, 25)  # Increased upper limit to show full peak gain
    axes[0].grid(True)
    
    # Plot MARS-processed gain
    axes[1].plot(mars_pattern.theta_angles, 
                mars_pattern.get_gain_db('e_co').values[freq_idx, :, phi_idx],
                label="MARS", color='red')
    axes[1].set_title("Gain After MARS")
    axes[1].set_xlabel("Theta (degrees)")
    axes[1].set_ylabel("Gain (dB)")
    axes[1].set_ylim(-40, 25)  # Increased upper limit to show full peak gain
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create an overlay plot showing both patterns
    plt.figure(figsize=(10, 6))
    
    plt.plot(original_pattern.theta_angles, 
            original_pattern.get_gain_db('e_co').values[freq_idx, :, phi_idx],
            label="Original")
    
    plt.plot(mars_pattern.theta_angles, 
            mars_pattern.get_gain_db('e_co').values[freq_idx, :, phi_idx],
            label="MARS", linestyle='--', color='red')
    
    plt.title("Comparison of Original and MARS-Processed Pattern")
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Gain (dB)")
    plt.ylim(-40, 25)  # Increased upper limit to show full peak gain
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """Demonstrate phase center finding and MARS application."""
    # Create the horn pattern
    original_pattern = create_horn_pattern()
    print(f"Created horn pattern with {original_pattern.polarization} polarization")
    
    # Apply phase shift to simulate offset phase center
    shifted_pattern = apply_phase_shift(original_pattern)
    
    # Visualize the phase shift
    visualize_phase(original_pattern, shifted_pattern)
    
    # Find phase center and correct
    corrected_pattern, phase_center = find_and_apply_phase_center(shifted_pattern)
    
    # Visualize the phase center correction
    visualize_phase_center_correction(shifted_pattern, corrected_pattern)
    
    # Apply MARS algorithm
    mars_pattern = apply_mars_algorithm(original_pattern)
    
    # Visualize the effect of MARS
    visualize_mars_effect(original_pattern, mars_pattern)
    
    print("\nPhase center and MARS processing complete!")
    
    plt.show()

if __name__ == "__main__":
    main()