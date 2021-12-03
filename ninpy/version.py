# /usr/bin/env python3
"""
"""
import importlib
import re
from typing import Tuple, Union

__all__ = ["extract_version", "get_version", "is_older_version"]


def extract_version(version: str) -> Tuple[int, int, int]:
    """Extract version information given a string in this format `r'[0-9].[0-9].[0-9]'`,
    r'[0-9].[0-9]', or r'[0-9]'. Note that for r'[0-9]' will return the first number in
    a given string.
    Example:
    >>> extract_version("1.9.1+cu111")
    ('1', '9', '1')
    >>> extract_version("1.9")
    ('1', '9')
    """
    versions = re.findall(r"[0-9].[0-9].[0-9]", version)
    if len(versions) == 0:
        versions = re.findall(r"[0-9].[0-9]", version)
        if len(versions) == 0:
            versions = re.findall(r"[0-9]", version)
            if len(versions) == 0:
                raise AssertionError(f"Cannot extract version from: {version}.")
            else:
                major = versions[0].split(".")
        else:
            major, minor = versions[0].split(".")
            patch = 0
    else:
        major, minor, patch = versions[0].split(".")
    return (int(major), int(minor), int(patch))


def get_version(module: str) -> Tuple[int, int, int]:
    """Extract version information given a module name.
    Expect module `__version__` is in this format `r'[0-9].[0-9].[0-9]'`.
    Example:
    >>> get_version("torch")
    ('1', '9', '1')
    """
    imported_module = importlib.import_module(module)
    versions = extract_version(imported_module.__version__)
    return versions


def is_older_version(
    module: str,
    version: Union[int, str, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
) -> bool:
    """Check local machine has a older version of `module` than a given `version` or not.
    Support formats of version in r`[0-9].[0-9].[0-9]`
    Example:
    >>> is_older_version("torch", "1.8.1")
    False
    >>> is_older_version("torch", "1.8")
    False
    >>> is_older_version("torch", (1, 8))
    False
    """
    assert isinstance(module, str)
    local_major, local_minor, local_patch = get_version(module)
    major = minor = patch = 0
    if isinstance(version, int):
        major = version
    elif isinstance(version, str):
        major, minor, patch = extract_version(version)
    elif isinstance(version, (tuple, list)):
        if len(version) == 1:
            major = version[0]
        elif len(version) == 2:
            major, minor = version
        elif len(version) == 3:
            major, minor, patch = version
        else:
            raise ValueError("Support only version with length 1, 2, and 3.")
    else:
        raise TypeError(f"{version} has a unsupported type: {type(version)}.")

    is_older = False
    if major > local_major:
        is_older = True
    elif major == local_major:
        if minor > local_minor:
            is_older = True
        elif major == local_major:
            if minor == local_minor:
                if patch > local_patch:
                    is_older = True
    return is_older
