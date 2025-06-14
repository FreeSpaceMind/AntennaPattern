# AntennaPattern Tutorial System

Welcome to the AntennaPattern tutorial system! This comprehensive guide will take you through all aspects of antenna pattern measurement, analysis, and processing using the AntennaPattern Python package.

## Prerequisites

Before starting these tutorials, ensure you have:

- Python 3.7 or higher installed
- The `antenna_pattern` package installed (`python install.py` from the root directory)
- Basic familiarity with Python and NumPy
- Understanding of antenna measurement concepts (recommended but not required)

## Data Files

The tutorials use example antenna measurement data located in the `examples/data/` directory:

- **`example_antenna.cut`** - NSI PLANET cut file format containing measured antenna pattern data
- **`example_antenna.ffd`** - Far-field data file with comprehensive measurement information

*Note: Place the example data files (`example_antenna.cut` and `example_antenna.ffd`) in the `examples/data/` directory before running the tutorials.*

## Learning Path

The tutorials are designed to be completed in sequence, building upon concepts from previous lessons:

### [Tutorial 1: Getting Started](./tutorial_01_getting_started.py) *(30 minutes)*
Learn the basics of loading antenna pattern data, understanding the AntennaPattern class, and basic visualization techniques.

**Topics covered:**
- Loading antenna pattern files
- Understanding pattern data structure
- Basic plotting and visualization
- Coordinate systems and units

### [Tutorial 2: File I/O and Data Formats](./tutorial_02_file_io.py) *(20 minutes)*
Master different file formats and data import/export capabilities.

**Topics covered:**
- Supported file formats (.cut, .ffd, .npz)
- File format conversion
- Data validation and error handling
- Metadata preservation

### [Tutorial 3: Measurement Corrections](./tutorial_03_corrections.py) *(45 minutes)*
Apply essential corrections to raw measurement data for accurate analysis.

**Topics covered:**
- Gain scaling and normalization
- Coordinate system transformations
- Pattern alignment and registration
- Measurement uncertainty handling

### [Tutorial 4: Dual-Sphere Processing](./tutorial_04_dual_sphere.py) *(45 minutes)*
Process dual-sphere measurements for enhanced pattern characterization.

**Topics covered:**
- Dual-sphere measurement principles
- Sphere-to-sphere transformations
- Pattern merging and blending
- Quality assessment metrics

### [Tutorial 5: Phase Center Analysis](./tutorial_05_phase_center.py) *(30 minutes)*
Determine and analyze antenna phase center characteristics.

**Topics covered:**
- Phase center definition and calculation
- Phase center optimization
- Impact on pattern measurements
- Validation techniques

### [Tutorial 6: MARS Filtering](./tutorial_06_mars.py) *(30 minutes)*
Apply Matrix Antenna Range Simulator (MARS) filtering for improved measurements.

**Topics covered:**
- MARS filtering principles
- Filter parameter selection
- Before/after comparison
- Performance evaluation

### [Tutorial 7: Analysis and Comparison](./tutorial_07_analysis.py) *(45 minutes)*
Perform comprehensive pattern analysis and comparison studies.

**Topics covered:**
- Pattern metrics calculation
- Multi-pattern comparison
- Statistical analysis
- Performance benchmarking

### [Tutorial 8: Complete Workflow](./tutorial_08_workflow.py) *(60 minutes)*
Integrate all concepts into a complete antenna pattern processing workflow.

**Topics covered:**
- End-to-end processing pipeline
- Automation and scripting
- Results documentation
- Best practices and troubleshooting

## Getting Help

- **Documentation**: Check the main package documentation and docstrings
- **Examples**: Review the `test_scripts/` directory for additional examples
- **Issues**: Report bugs or questions on the project repository

## Estimated Total Time: 5 hours

Each tutorial includes:
- Clear learning objectives
- Step-by-step code examples
- Practical exercises
- Troubleshooting tips
- References for further reading

Start with Tutorial 1 and work through the sequence at your own pace. Each tutorial builds upon the previous one, so completing them in order is recommended for the best learning experience.