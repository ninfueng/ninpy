#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import subprocess
import sys


def run_python(cmd: str, getline: int = -2) -> float:
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
    return float(stdout.split("\n")[getline])
