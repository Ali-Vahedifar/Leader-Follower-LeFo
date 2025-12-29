#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leader-Follower (LeFo): Signal Prediction for Loss Mitigation in Tactile Internet
A Leader-Follower Game-Theoretic Approach

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering, Aarhus University, Denmark

This paper has been accepted to IEEE MLSP 2025 (Istanbul, Turkey)

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lefo-tactile",
    version="1.0.0",
    author="Mohammad Ali Vahedifar",
    author_email="av@ece.au.dk",
    description="Leader-Follower Game-Theoretic Signal Prediction for Tactile Internet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ali-Vahedifar/Leader-Follower-LeFo",
    project_urls={
        "Bug Tracker": "https://github.com/Ali-Vahedifar/Leader-Follower-LeFo/issues",
        "Documentation": "https://github.com/Ali-Vahedifar/Leader-Follower-LeFo#readme",
        "Paper": "https://ieeexplore.ieee.org/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lefo-train=lefo.cli:train",
            "lefo-evaluate=lefo.cli:evaluate",
            "lefo-predict=lefo.cli:predict",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "tactile-internet",
        "signal-prediction",
        "game-theory",
        "stackelberg-game",
        "neural-network",
        "haptic-feedback",
        "teleoperation",
        "human-robot-interaction",
    ],
)
