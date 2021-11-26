#!/usr/bin/env python3
"""YAML or json related .
@author: Ninnart Fuengfusin"""
import os
import json
import pickle
from typing import Any, Dict

import yaml

__all__ = [
    "load_pickle",
    "dump_pickle",
    "load_yaml",
    "dump_yaml",
    "load_json",
    "dump_json",
]


def load_pickle(pickle_path: str) -> Any:
    assert isinstance(pickle_path, str)
    pickle_path = os.path.expanduser(pickle_path)
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def dump_pickle(save_path: str, obj: Any) -> None:
    """Save object `obj` as a pickle file."""
    assert isinstance(save_path, str)
    save_path = os.path.expanduser(save_path)
    with open(save_path, "wb") as p:
        pickle.dump(obj, p, protocol=pickle.HIGHEST_PROTOCOL)


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load a yaml file.
    Example:
    >>> load_yaml('./config.yaml')
    """
    assert isinstance(yaml_path, str)
    yaml_path = os.path.expanduser(yaml_path)
    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def dump_yaml(input: Dict[str, Any], save_path: str) -> None:
    """Refer: https://stackabuse.com/reading-and-writing-yaml-to-a-file-in-python/
    Designed to dump hyper parameters or results as yaml file.
    TODO: update to support automatically convert object type to python type.
    Example:
    >>> input = load_yaml('./config.yaml')
    >>> dump_yaml(input, './config2.yaml')
    """
    assert isinstance(save_path, str)
    assert isinstance(input, dict)
    save_path = os.path.expanduser(save_path)
    with open(save_path, "w") as f:
        yaml.dump(input, f)


def load_json(json_path: str) -> Dict[str, Any]:
    assert isinstance(json_path, str)
    json_path = os.path.expanduser(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def dump_json(input: Dict[str, Any], save_path: str, indent: int = 4) -> None:
    assert isinstance(save_path, str)
    assert isinstance(indent, int)
    assert isinstance(input, dict)
    with open(save_path, "w") as f:
        json.dump(input, f, indent=indent)
