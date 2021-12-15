#!/usr/bin/env python3
import os

from setuptools import find_packages, setup


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ninpy",
    version="0.0.4",
    author="Ninnart Fuengfusin",
    author_email="ninnart.fuengfusin@yahoo.com",
    description="Collection of reuse-able modules designed by Ninnart Fuengfusin.",
    license="MIT",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    classifier=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
