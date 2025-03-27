#!/usr/bin/env python
"""
Installation script for antenna_pattern package.
This script installs the package in development mode and runs tests.
"""

import os
import subprocess
import sys

def main():
    """Main installation function."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("===== Installing antenna_pattern package =====")
    print(f"Installation directory: {script_dir}")
    
    # Check if src directory exists
    src_dir = os.path.join(script_dir, "src")
    if not os.path.isdir(src_dir):
        print(f"✗ Error: Could not find src directory at {src_dir}")
        print("  Make sure you're running this script from the project root.")
        return 1
    
    # Install in development mode
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=script_dir)
        print("✓ Installation successful")
    except subprocess.CalledProcessError:
        print("✗ Installation failed")
        return 1
    
    # Run tests
    print("\n===== Running tests =====")
    try:
        # Add src directory to PYTHONPATH for tests
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{python_path}"
        else:
            env["PYTHONPATH"] = src_dir
            
        subprocess.check_call([sys.executable, "-m", "pytest", "tests"], cwd=script_dir, env=env)
        print("✓ Tests passed")
    except subprocess.CalledProcessError:
        print("✗ Some tests failed")
        return 1
    
    print("\n===== Installation complete =====")
    print("antenna_pattern is now installed in development mode.")
    print("You can import it in your Python code with: import antenna_pattern")
    return 0

if __name__ == "__main__":
    sys.exit(main())