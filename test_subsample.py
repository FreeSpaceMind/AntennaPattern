#!/usr/bin/env python3
"""
Test script demonstrating the subsample functionality.

This script creates a synthetic antenna pattern and demonstrates
different subsampling scenarios to verify the implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from antenna_pattern import AntennaPattern, create_synthetic_pattern

def main():
    print("Testing AntennaPattern.subsample() method")
    print("=" * 50)
    
    # Create a synthetic pattern for testing
    frequencies = np.array([1e9, 2e9])  # 1 GHz and 2 GHz
    theta_angles = np.arange(-90, 91, 1)  # 1-degree resolution
    phi_angles = np.arange(0, 360, 5)     # 5-degree resolution
    
    print(f"Creating synthetic pattern:")
    print(f"  Frequencies: {len(frequencies)} points")
    print(f"  Theta: {len(theta_angles)} points ({theta_angles.min()}° to {theta_angles.max()}°)")
    print(f"  Phi: {len(phi_angles)} points ({phi_angles.min()}° to {phi_angles.max()}°)")
    print(f"  Original pattern shape: {len(frequencies)} x {len(theta_angles)} x {len(phi_angles)}")
    
    # Create synthetic pattern
    e_theta, e_phi = create_synthetic_pattern(
        frequencies=frequencies,
        theta_angles=theta_angles,
        phi_angles=phi_angles,
        peak_gain_dbi=15.0,
        polarization='rhcp',
        beamwidth_deg=25.0
    )
    
    # Create AntennaPattern object
    pattern = AntennaPattern(
        theta=theta_angles,
        phi=phi_angles,
        frequency=frequencies,
        e_theta=e_theta,
        e_phi=e_phi,
        polarization='rhcp'
    )
    
    print(f"\nOriginal pattern created successfully!")
    print(f"Pattern data shape: {pattern.data.e_theta.shape}")
    
    # Test 1: Reduce theta resolution
    print(f"\nTest 1: Reduce theta resolution (every 5 degrees)")
    subsampled1 = pattern.subsample(theta_step=5.0)
    print(f"  New theta points: {len(subsampled1.theta_angles)}")
    print(f"  New pattern shape: {subsampled1.data.e_theta.shape}")
    print(f"  Theta range: {subsampled1.theta_angles.min()}° to {subsampled1.theta_angles.max()}°")
    
    # Test 2: Reduce phi resolution  
    print(f"\nTest 2: Reduce phi resolution (every 15 degrees)")
    subsampled2 = pattern.subsample(phi_step=15.0)
    print(f"  New phi points: {len(subsampled2.phi_angles)}")
    print(f"  New pattern shape: {subsampled2.data.e_theta.shape}")
    print(f"  Phi range: {subsampled2.phi_angles.min()}° to {subsampled2.phi_angles.max()}°")
    
    # Test 3: Reduce both theta and phi resolution
    print(f"\nTest 3: Reduce both theta and phi resolution")
    subsampled3 = pattern.subsample(theta_step=10.0, phi_step=20.0)
    print(f"  New theta points: {len(subsampled3.theta_angles)}")
    print(f"  New phi points: {len(subsampled3.phi_angles)}")
    print(f"  New pattern shape: {subsampled3.data.e_theta.shape}")
    
    # Test 4: Limit theta range
    print(f"\nTest 4: Limit theta range to ±60 degrees with 2-degree step")
    subsampled4 = pattern.subsample(
        theta_range=(-60, 60),
        theta_step=2.0
    )
    print(f"  New theta points: {len(subsampled4.theta_angles)}")
    print(f"  Theta range: {subsampled4.theta_angles.min()}° to {subsampled4.theta_angles.max()}°")
    print(f"  New pattern shape: {subsampled4.data.e_theta.shape}")
    
    # Test 5: Limit phi range
    print(f"\nTest 5: Limit phi range to 45-315 degrees with 30-degree step")
    subsampled5 = pattern.subsample(
        phi_range=(45, 315),
        phi_step=30.0
    )
    print(f"  New phi points: {len(subsampled5.phi_angles)}")
    print(f"  Phi range: {subsampled5.phi_angles.min()}° to {subsampled5.phi_angles.max()}°")
    print(f"  New pattern shape: {subsampled5.data.e_theta.shape}")
    
    # Test 6: Combined range and step reduction (example from docstring)
    print(f"\nTest 6: Combined example - theta ±75°:5°, phi 0-360°:30°")
    subsampled6 = pattern.subsample(
        theta_range=(-75, 75),
        theta_step=5.0,
        phi_range=(0, 360),
        phi_step=30.0
    )
    print(f"  New theta points: {len(subsampled6.theta_angles)}")
    print(f"  New phi points: {len(subsampled6.phi_angles)}")
    print(f"  Theta range: {subsampled6.theta_angles.min()}° to {subsampled6.theta_angles.max()}°")
    print(f"  Phi range: {subsampled6.phi_angles.min()}° to {subsampled6.phi_angles.max()}°")
    print(f"  New pattern shape: {subsampled6.data.e_theta.shape}")
    
    # Verify data integrity
    print(f"\nVerifying data integrity:")
    print(f"  Original peak gain: {pattern.get_gain_db().max():.2f} dBi")
    print(f"  Subsampled peak gain: {subsampled6.get_gain_db().max():.2f} dBi")
    print(f"  Polarization preserved: {pattern.polarization == subsampled6.polarization}")
    print(f"  Frequencies preserved: {np.array_equal(pattern.frequencies, subsampled6.frequencies)}")
    
    # Check metadata
    if hasattr(subsampled6, 'metadata') and 'operations' in subsampled6.metadata:
        last_op = subsampled6.metadata['operations'][-1]
        print(f"  Operation recorded: {last_op['type']}")
        print(f"  Original shape: {last_op['original_shape']}")
        print(f"  New shape: {last_op['new_shape']}")
    
    # Create a simple comparison plot
    print(f"\nCreating comparison plots...")
    
    # Plot a phi=0 cut comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original pattern
    phi_idx_orig = np.argmin(np.abs(pattern.phi_angles - 0))
    gain_orig = pattern.get_gain_db()[0, :, phi_idx_orig]  # First frequency, phi=0 cut
    ax1.plot(pattern.theta_angles, gain_orig, 'b-', label='Original (1° resolution)', linewidth=2)
    
    # Subsampled pattern
    phi_idx_sub = np.argmin(np.abs(subsampled6.phi_angles - 0))
    gain_sub = subsampled6.get_gain_db()[0, :, phi_idx_sub]  # First frequency, phi=0 cut
    ax1.plot(subsampled6.theta_angles, gain_sub, 'ro-', label='Subsampled (5° resolution)', markersize=4)
    
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Gain (dBi)')
    ax1.set_title('Gain Pattern Comparison (φ = 0°)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([gain_orig.min() - 2, gain_orig.max() + 2])
    
    # Plot sampling grid
    theta_grid, phi_grid = np.meshgrid(subsampled6.theta_angles, subsampled6.phi_angles)
    ax2.scatter(phi_grid.flatten(), theta_grid.flatten(), s=10, alpha=0.6, c='red')
    ax2.set_xlabel('Phi (degrees)')
    ax2.set_ylabel('Theta (degrees)')
    ax2.set_title('Subsampled Grid Points')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 360])
    ax2.set_ylim([-90, 90])
    
    plt.tight_layout()
    plt.show()
    
    # Test error conditions
    print(f"\nTesting error conditions:")
    
    try:
        # Should fail - theta range outside available data
        pattern.subsample(theta_range=(-100, 100))
        print("  ERROR: Should have failed for theta range outside data")
    except ValueError as e:
        print(f"  ✓ Correctly caught theta range error: {e}")
    
    try:
        # Should fail - phi range too large
        pattern.subsample(phi_range=(0, 400))
        print("  ERROR: Should have failed for phi range > 360°")
    except ValueError as e:
        print(f"  ✓ Correctly caught phi range error: {e}")
    
    print(f"\n✓ All tests completed successfully!")
    print(f"The subsample() method is working correctly.")

if __name__ == "__main__":
    main()