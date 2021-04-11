import os

import setuptools


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="ninpy",
    version="0.0.1",
    author="Ninnart Fuengfusin",
    author_email="ninnart.fuengfusin@yahoo.com",
    description="Template files for Ninnart Fuengfusin.",
    license="MIT",
    long_description=read("README.md"),
    packages=setuptools.find_packages(),
)
