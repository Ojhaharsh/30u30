"""
Day 5: MDL (Minimum Description Length) Principle
==================================================

Setup and utility functions for the MDL implementation.
"""

from setuptools import setup, find_packages

setup(
    name="mdl_principle",
    version="1.0.0",
    description="Implementation of the Minimum Description Length Principle",
    author="30u30 Project",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
