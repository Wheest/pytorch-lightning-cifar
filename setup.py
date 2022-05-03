#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="pytorch_lightning_cifar",
    version="0.1.1",
    url="https://github.com/Wheest/pytorch-lightning-cifar.git",
    author="Perry Gibson",
    author_email="perry@gibsonic.org",
    description="Common CNN models defined for PyTorch Lightning ",
    packages=find_packages(),
    install_requires=[
        "pytorch_lightning>=1.6.2",
        "torchvision >= 0.12.0",
        "lightning-bolts >= 0.5.0",
    ],
)
