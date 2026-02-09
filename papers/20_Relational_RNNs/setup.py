from setuptools import setup, find_packages

setup(
    name="relational_rnns",
    version="0.1.0",
    description="PyTorch implementation of Relational Recurrent Neural Networks (Santoro et al., 2018)",
    author="30u30 Project",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
