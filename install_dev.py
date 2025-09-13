#!/usr/bin/env python3
"""
Development installation script for the Paired Replacement Algorithm package.

This script installs the package in development mode so imports work cleanly.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Install the package in development mode."""
    project_root = Path(__file__).parent
    
    print("Installing Paired Replacement Algorithm package in development mode...")
    
    try:
        # Install in development mode
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=project_root, check=True, capture_output=True, text=True)
        
        print("✅ Package installed successfully!")
        print("\nNow you can:")
        print("  • Import: from paired_replacement.src.python.paired_cache import PairedCache")
        print("  • Run scripts from anywhere in the project")
        print("  • Use console scripts: run-microbench, run-autotune, etc.")
        
    except subprocess.CalledProcessError as e:
        print("❌ Installation failed:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()