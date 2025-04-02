"""
Example: Saving and loading antenna patterns in various file formats.

This example demonstrates how to:
1. Create a synthetic horn antenna pattern
2. Save to NPZ format (native format)
3. Save to CUT format 
4. Load from NPZ format
5. Load from CUT format
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path if not already installed
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from antenna_pattern import AntennaPattern
from antenna_pattern.ant_io import save_pattern_npz, load_pattern_npz, read_cut

# Import the function to create our test pattern
from create_pattern_util import create_horn_pattern, visualize_pattern

def save_and_load_npz(pattern, output_dir):
    """
    Save pattern to NPZ and load it back.
    
    Args:
        pattern: AntennaPattern to save
        output_dir: Directory to save files
    
    Returns:
        AntennaPattern: Loaded pattern
    """
    # Create metadata
    metadata = {
        "antenna_type": "Horn",
        "description": "Synthetic horn pattern with 1dB axial ratio",
        "created_by": "antenna_pattern example script",
        "notes": "This is an example pattern for demonstration purposes"
    }
    
    # Save pattern to NPZ
    npz_path = os.path.join(output_dir, "horn_pattern.npz")
    save_pattern_npz(pattern, npz_path, metadata)
    print(f"Pattern saved to {npz_path}")
    
    # Load pattern from NPZ
    loaded_pattern, loaded_metadata = load_pattern_npz(npz_path)
    print("Pattern loaded from NPZ with metadata:")
    for key, value in loaded_metadata.items():
        print(f"  {key}: {value}")
    
    return loaded_pattern

def save_and_load_cut(pattern, output_dir):
    """
    Save pattern to CUT format and load it back.
    
    Args:
        pattern: AntennaPattern to save
        output_dir: Directory to save files
    
    Returns:
        AntennaPattern: Loaded pattern
    """
    # Save pattern to CUT format with different polarization formats
    for pol_format, pol_name in [(1, "theta_phi"), (2, "rhcp_lhcp"), (3, "ludwig3")]:
        cut_path = os.path.join(output_dir, f"horn_pattern_{pol_name}.cut")
        pattern.write_cut(cut_path, polarization_format=pol_format)
        print(f"Pattern saved to {cut_path} with polarization format {pol_format}")
    
    # Load pattern from CUT format (using the theta_phi format)
    cut_path = os.path.join(output_dir, "horn_pattern_theta_phi.cut")
    loaded_pattern = read_cut(cut_path, 8e9, 12e9)
    print(f"Pattern loaded from {cut_path}")
    
    return loaded_pattern

def main():
    """Demonstrate saving and loading antenna patterns."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the horn pattern
    print("Creating horn antenna pattern...")
    pattern = create_horn_pattern()
    
    # Save and load using NPZ format
    print("\nTesting NPZ format...")
    loaded_npz = save_and_load_npz(pattern, output_dir)
    visualize_pattern(loaded_npz, title="Pattern Loaded from NPZ")
    
    # Save and load using CUT format
    print("\nTesting CUT format...")
    loaded_cut = save_and_load_cut(pattern, output_dir)
    visualize_pattern(loaded_cut, title="Pattern Loaded from CUT")
    
    print("\nSaving and loading complete! Files are located in:", output_dir)

if __name__ == "__main__":
    main()