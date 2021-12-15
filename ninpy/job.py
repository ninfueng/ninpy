#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import subprocess
import sys
from typing import Optional


def run_python(cmd: str, getline: int = -2) -> Optional[float]:
    """Run command and receive a stdout `getline` line.
    Example:
    >>> run_python("python main.py")
    """
    assert isinstance(cmd, str)
    assert isinstance(getline, int)
    PYTHON = sys.executable
    if cmd.find("python3") > -1:
        cmd = cmd.replace("python3", PYTHON)
    elif cmd.find("python") > -1:
        cmd = cmd.replace("python", PYTHON)
    else:
        raise NotImplementedError()
    cmd = cmd.split(" ")
    stdout = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode()
    try:
        result = float(stdout.split("\n")[getline])
    except (IndexError, ValueError):
        # Catche errors when the float converting is not possible or
        # Number of line from stdout contains lower number of lines than `getline`.
        result = None
    return result
