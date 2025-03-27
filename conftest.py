"""
Test configuration file for pytest.
This file helps pytest discover the src module.
"""
import os
import sys

# Add the src directory to the path for test discovery
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)