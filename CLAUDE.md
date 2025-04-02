# CLAUDE.md - Usage Guidelines for this Repository

## Build & Test Commands
- Install package: `python install.py`
- Install dev dependencies: `pip install -e .[dev]`
- Run all tests: `python -m pytest tests`
- Run single test: `python -m pytest tests/test_antenna_pattern.py::test_function_name`
- Format code: `black src tests`
- Sort imports: `isort src tests`
- Lint code: `flake8 src tests`

## Code Style Guidelines
- **Formatting**: Black (line-length=88)
- **Imports**: isort with black profile (multi_line_output=3)
- **Types**: Use type hints throughout the code
- **Naming**:
  - Classes: CamelCase (AntennaPattern)
  - Functions/Methods: snake_case (get_gain_db)
  - Constants: UPPERCASE (VALID_POLARIZATIONS)
  - Private attributes: prefix with underscore (_cache)
- **Error handling**: Use specific exception types with clear error messages
- **Documentation**: Use docstrings with argument descriptions and return types
- **Dependencies**: Maintain compatibility with numpy, scipy, xarray, matplotlib