from setuptools import setup, find_packages

setup(
    name="coffee-automaton",
    version="0.1.0",
    description="Understanding Complexity Theory through Coffee Automaton",
    author="30u30 Learning Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "jupyter>=1.0.0",
        "pillow>=8.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
