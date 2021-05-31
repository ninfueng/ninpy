#!/usr/bin/env python3
"""@author: Ninnart Fuengfusin"""
import sys
import time
from typing import Any, Dict

import yaml

from ninpy.data import AttrDict


def load_yaml(yaml_file: str, with_attrdict: bool = True) -> dict:
    """Load a yaml file with an option to load into AttrDict or not.
    Example:
    >>> load_yaml('./config.yaml')
    """
    assert isinstance(yaml_file, str)
    assert isinstance(with_attrdict, bool)
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if with_attrdict:
        data = AttrDict(data)
    return data


def dump_yaml(input: dict, save_loc: str) -> None:
    """Refer: https://stackabuse.com/reading-and-writing-yaml-to-a-file-in-python/
    Designed to dump hyper parameters or results as yaml file.
    TODO: update to support automatically convert object type to python type.
    Example:
    >>> input = load_yaml('./config.yaml')
    >>> dump_yaml(input, './config2.yaml')
    """
    assert isinstance(save_loc, str)
    assert isinstance(input, dict)
    with open(save_loc, "w") as f:
        yaml.dump(input, f)


def dict2str(input: Dict[str, Any]) -> str:
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
    string = ""
    for k, v in sorted(input.items()):
        if not isinstance(v, dict):
            string += f"{k}:{v}-"
        else:
            string += f"{k}::"
            string += dict2str(v)
    return string


def args2str(args) -> str:
    r"""Generate a string from argparse parser.
    If argparse records hyper-parameters, then this compresses
    all hyper-parameters to a string.
    """
    # args = parser.parse_args()
    args_dict = vars(args)
    string = dict2str(args_dict)
    return string


def args2yaml(args, yaml_name: str) -> None:
    r"""Convert argparse to a yaml file."""
    args = vars(args)
    dump_yaml(args, yaml_name)


def name_experiment(hparams: dict) -> str:
    """Combine all hparams into name and make sure save-able in a folder."""
    exp_pth = dict2str(hparams)
    datetime = time.strftime("%Y:%m:%d-%H:%M:%S")
    exp_pth = f"{datetime}-{exp_pth}"
    # Protect for an argument with path and /, dash.
    # FileNotFoundError: [Errno 2] No such file or directory:
    exp_pth = exp_pth.replace("/", "_")

    if sys.platform == "win32":
        # Detect OSError: [WinError 123] The filename,
        # directory name, or volume label syntax is incorrect:
        exp_pth = exp_pth.replace(".", "_")
        exp_pth = exp_pth.replace(":", "_")
    if len(exp_pth) > 255:
        exp_pth = exp_pth[0:254]
    return exp_pth
