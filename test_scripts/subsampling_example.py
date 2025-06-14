#!/usr/bin/env python
"""
Test script for AntennaPattern.subsample() method.

This script loads a large .cut file and reduces it to smaller dimensions
suitable for tutorial examples.
"""

import os
from antenna_pattern import read_cut, read_ffd
import numpy as np

def main():
    # Configuration
    input_file = "test_scripts\input.cut"  # Update this path
    output_file = "sample_antenna.cut"
    
    # Original file specs (update these to match your file)
    freq_start = 3.15e9  # Hz
    freq_end = 3.25e9    # Hz
    
    # Target dimensions
    target_theta_range = (-150, 150)  # degrees
    target_theta_step = 2.0           # degrees
    target_phi_range = (0, 345)       # degrees  
    target_phi_step = 15.0            # degrees
    
    print("=" * 60)
    print("AntennaPattern Subsampling Test")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please update the 'input_file' path in the script.")
        return
    
    # Load original pattern
    print(f"Loading original pattern from: {input_file}")
    try:
        original_pattern = read_cut(input_file, freq_start, freq_end)
        print("✓ Original pattern loaded successfully")
    except Exception as e:
        print(f"ERROR loading pattern: {e}")
        return
    
    # Display original pattern info
    print(f"\nOriginal Pattern Dimensions:")
    print(f"  Frequencies: {len(original_pattern.frequencies)} points ({freq_start/1e9:.2f} - {freq_end/1e9:.2f} GHz)")
    print(f"  Theta: {len(original_pattern.theta_angles)} points ({original_pattern.theta_angles.min():.1f}° to {original_pattern.theta_angles.max():.1f}°)")
    print(f"  Phi: {len(original_pattern.phi_angles)} points ({original_pattern.phi_angles.min():.1f}° to {original_pattern.phi_angles.max():.1f}°)")
    print(f"  Polarization: {original_pattern.polarization}")
    
    # Calculate original file size
    original_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"  File size: {original_size_mb:.1f} MB")
    
    # Perform subsampling
    print(f"\nPerforming subsampling...")
    print(f"  Target theta: {target_theta_range[0]}° to {target_theta_range[1]}° step {target_theta_step}°")
    print(f"  Target phi: {target_phi_range[0]}° to {target_phi_range[1]}° step {target_phi_step}°")
    
    try:
        subsampled_pattern = original_pattern.subsample(
            theta_range=target_theta_range,
            theta_step=target_theta_step,
            phi_range=target_phi_range,
            phi_step=target_phi_step
        )
        print("✓ Subsampling completed successfully")
    except Exception as e:
        print(f"ERROR during subsampling: {e}")
        return
    
    # Display subsampled pattern info
    print(f"\nSubsampled Pattern Dimensions:")
    print(f"  Frequencies: {len(subsampled_pattern.frequencies)} points (unchanged)")
    print(f"  Theta: {len(subsampled_pattern.theta_angles)} points ({subsampled_pattern.theta_angles.min():.1f}° to {subsampled_pattern.theta_angles.max():.1f}°)")
    print(f"  Phi: {len(subsampled_pattern.phi_angles)} points ({subsampled_pattern.phi_angles.min():.1f}° to {subsampled_pattern.phi_angles.max():.1f}°)")
    print(f"  Polarization: {subsampled_pattern.polarization} (unchanged)")
    
    # Calculate reduction ratios
    theta_reduction = len(original_pattern.theta_angles) / len(subsampled_pattern.theta_angles)
    phi_reduction = len(original_pattern.phi_angles) / len(subsampled_pattern.phi_angles)
    total_reduction = theta_reduction * phi_reduction
    
    print(f"\nReduction Ratios:")
    print(f"  Theta points: {theta_reduction:.1f}x reduction")
    print(f"  Phi points: {phi_reduction:.1f}x reduction") 
    print(f"  Total data: {total_reduction:.1f}x reduction")
    print(f"  Expected file size: ~{original_size_mb/total_reduction:.1f} MB")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created directory: {output_dir}")
    
    # Save subsampled pattern
    print(f"\nSaving subsampled pattern to: {output_file}")
    try:
        subsampled_pattern.write_cut(output_file)
        print("✓ Subsampled pattern saved successfully")
    except Exception as e:
        print(f"ERROR saving pattern: {e}")
        return
    
    # Verify saved file
    if os.path.exists(output_file):
        actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Actual file size: {actual_size_mb:.1f} MB")
        print(f"  Size reduction: {original_size_mb/actual_size_mb:.1f}x")
        
        # Quick validation by reloading
        print(f"\nValidating saved file...")
        try:
            validation_pattern = read_cut(output_file, freq_start, freq_end)
            print("✓ Saved file loads correctly")
            
            # Check dimensions match
            if (len(validation_pattern.theta_angles) == len(subsampled_pattern.theta_angles) and
                len(validation_pattern.phi_angles) == len(subsampled_pattern.phi_angles)):
                print("✓ Dimensions match expected values")
            else:
                print("⚠ Dimension mismatch detected")
                
        except Exception as e:
            print(f"ERROR validating saved file: {e}")
    
    print(f"\n" + "=" * 60)
    print("Subsampling test completed!")
    print(f"Tutorial-ready file created: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()