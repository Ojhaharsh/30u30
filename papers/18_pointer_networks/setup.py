from setuptools import setup, find_packages

setup(
    name="day18_pointer_networks",
    version="1.0.0",
    description="Pointer Networks implementation for 30u30 project",
    author="30u30 Project",
    author_email="project@30u30.ai",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
