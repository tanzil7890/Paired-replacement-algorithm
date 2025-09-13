#!/usr/bin/env python3
"""
Setup script for Paired Replacement Algorithm research project.

This allows proper Python package imports and development installation.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README file
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="paired-replacement-algorithm",
    version="1.0.0",
    author="Mohammad Tanzil Idrisi",
    description="A novel cache replacement technique for dynamic sparse neural network inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.12.0",
    ],
    extras_require={
        "benchmarks": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
        ],
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "run-microbench=benchmarks.scripts.run_microbench:main",
            "run-microbench-torch=benchmarks.scripts.run_microbench_torch:main",
            "run-autotune=benchmarks.scripts.run_autotune:main",
        ],
    },
)