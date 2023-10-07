#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="GCPNet-EMA",
    version="0.0.1",
    description="Source code for the paper 'Protein Structure Accuracy Estimation using Geometry-Complete Perceptron Networks'.",
    author="Alex Morehead",
    author_email="acmwhb@umsystem.edu",
    url="https://github.com/BioinfoMachineLearning/GCPNet-EMA",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
