# Antenna Pattern Examples

This directory contains examples for the `antenna_pattern` library demonstrating various antenna pattern operations.

## Overview of Examples

1. **create_pattern_util.py** - Utility module that creates synthetic patterns using the `create_synthetic_pattern` function
2. **synthetic_pattern_example.py** - Demonstrates creating patterns with high-level parameters
3. **saving_and_loading.py** - Demonstrates saving and loading patterns in various file formats (NPZ, CUT)
4. **polarization_conversion.py** - Shows how to convert between different polarization types (RHCP/LHCP, X/Y, Theta/Phi)
5. **phase_center_and_mars.py** - Demonstrates phase center finding, phase shifting, and MARS algorithm application
6. **gain_scaling.py** - Shows different methods to scale pattern gain (uniform, frequency-dependent, 2D)

## Running the Examples

To run any example, ensure the package is installed or the `src` directory is in your Python path. Then run:

```bash
python example_tests/example_name.py
```

For example:

```bash
python example_tests/synthetic_pattern_example.py
```

## Example Outputs

All examples generate visualization plots using matplotlib's `plt.show()` function.

The saving_and_loading.py example creates pattern files in different formats in the `output` directory.

## Dependencies

The examples require the same dependencies as the main package:
- numpy
- scipy
- xarray
- matplotlib

## Using as Learning Resources

These examples are designed to demonstrate the capabilities of the `antenna_pattern` library and serve as references for common use cases. Feel free to modify them to learn how different parameters affect the results.

For a structured walkthrough, it's recommended to explore the examples in this order:
1. synthetic_pattern_example.py
2. saving_and_loading.py
3. polarization_conversion.py
4. phase_center_and_mars.py
5. gain_scaling.py