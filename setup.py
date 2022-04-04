import os.path as osp

from setuptools import find_packages, setup

requirements = []

__version__ = "0.0.1"

setup(
    name="fabricflownet",
    version=__version__,
    author="Thomas Weng",
    packages=find_packages(),
    install_requires=requirements,
)
