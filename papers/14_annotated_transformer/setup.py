"""
Setup script for The Annotated Transformer (Day 14)
===================================================

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="annotated-transformer",
    version="1.0.0",
    description="Production PyTorch Transformer implementation from The Annotated Transformer",
    author="30u30 Project",
    url="https://github.com/Ojhaharsh/30u30",
    
    py_modules=["implementation", "visualization", "train_minimal"],
    
    python_requires=">=3.8",
    
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "translation": [
            "tokenizers>=0.10.0",
            "sacrebleu>=2.0.0",
            "sentencepiece>=0.1.95",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    keywords="transformer attention nlp deep-learning pytorch",
)
