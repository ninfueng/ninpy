import os
from setuptools import setup

def read(fname: str) -> str:
    assert isinstance(fname, str)
    return open(
        os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = 'ninpy',
    version = '0.0.1',
    author = 'Ninnart Fuengfusin',
    author_email = "ninnart.fuengfusin@yahoo.com",
    description = 'Template files designed for Ninnart Fuengfusin.',
    license = 'MIT',
    long_description=read('README.md'),
)