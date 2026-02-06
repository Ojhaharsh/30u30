from setuptools import setup, find_packages

setup(
    name="ntm_pytorch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
