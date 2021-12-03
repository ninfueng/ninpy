#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import argparse
import glob
import os
import shutil
import sys
from functools import reduce
from typing import Any, Dict, List

from ninpy.config import dump_json, dump_yaml


def dict2str(input: Dict[str, Any], sort: bool = True) -> str:
    """Given a dict recursively includes all parameters into a string.
    `::` subdict, `:` dict, and - next var.
    Example:
    >>> v = dict2str({1:2, 3:4, 5:{6:7}, 8:{9:{10:11}}})
    '1:2-3:4-5::6:7-8::9::10:11-'
    Args:
        input (dict): dict to reduce to a string.
    Return:
        string (str): compressed string.
    """
    items = input.items()
    if sort:
        items = sorted(input.items())

    string = ""
    for k, v in items:
        if not isinstance(v, dict):
            string += f"{k}:{v}-"
        else:
            string += f"{k}::"
            string += dict2str(v)
    return string


def args2str(args: argparse.Namespace) -> str:
    """Generate a string from argparse parser.
    If argparse records hyper-parameters, then this compresses
    all hyper-parameters to a string.
    """
    args_dict = vars(args)
    string = dict2str(args_dict)
    return string


def args2yaml(args: argparse.Namespace, yaml_name: str) -> None:
    """Convert argparse to a dict and save as a yaml file."""
    args = vars(args)
    dump_yaml(args, yaml_name)


def args2json(args: argparse.Namespace, json_name: str) -> None:
    """Convert argparse to a dict and save as a json file."""
    args = vars(args)
    dump_json(args, json_name)


def name_experiment(hparams: Dict[str, Any]) -> str:
    """Combine all hparams into name and make sure save-able in a folder."""
    exp_path = dict2str(hparams)
    datetime = get_datetime()
    exp_path = f"{datetime}-{exp_pth}"
    # Protect for an argument with path and /, dash.
    # FileNotFoundError: [Errno 2] No such file or directory:
    exp_path = exp_path.replace("/", "_")

    if sys.platform == "win32":
        # Detect OSError: [WinError 123] The filename,
        # directory name, or volume label syntax is incorrect:
        exp_path = exp_path.replace(".", "_")
        exp_path = exp_path.replace(":", "_")
    if len(exp_path) > 255:
        exp_path = exp_path[0:254]
    return exp_path


def set_experiment(
    exp_path: str,
    save_list: List[str] = ["*.py", "*.sh", "*.yaml", "*.json"],
    rm_exist: bool = True,
) -> str:
    """Create a folder with the name `exp_path`.
    Copy all files that can be glob from `save_list` into `exp_path/scripts/`.
    With `save_list = ["*.py", "*.sh"]`, the folder can be as follows:
    exp_path
     └── scripts
         ├── *.py
         ├── ...
         └── *.sh
    Example:
    >>> set_experiment("experiment")
    """
    assert isinstance(exp_path, str)
    assert len(save_list) > 0

    if rm_exist and os.path.exists(exp_path):
        shutil.rmtree(exp_path)

    if not os.path.exists(exp_path):
        if sys.platform == "win32":
            # For OSError: [WinError 123] The filename,
            # directory name, or volume label syntax is incorrect:
            exp_path = exp_path.replace(".", "_")
            exp_path = exp_path.replace(":", "_")
        os.mkdir(exp_path)
        os.makedirs(os.path.join(exp_path, "scripts"), exist_ok=True)

    save_scripts = [glob.glob(t, recursive=False) for t in save_list]
    save_scripts = reduce(lambda x, y: x + y, save_scripts)
    for s in save_scripts:
        dest_path = os.path.join(exp_path, "scripts", str(s))
        shutil.copyfile(s, dest_path)
    return exp_path


if __name__ == "__main__":
    set_experiment("experiment", ["*.py", "*yaml"])
