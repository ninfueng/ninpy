#!/usr/bin/env python3
"""Config file for `yaml` or `json` related .
@author: Ninnart Fuengfusin"""
import argparse
import json
import logging
import os
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
    "json2argparse",
]
logger = logging.getLogger("ninpy")


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
    """"""
    assert isinstance(json_path, str)
    json_path = os.path.expanduser(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def dump_json(input: Dict[str, Any], save_path: str, indent: int = 4) -> None:
    """Save a dict to the json file.
    Example:
    >>> dump_json(input)
    """
    assert isinstance(save_path, str)
    assert isinstance(indent, int)
    assert isinstance(input, dict)
    save_path = os.path.expanduser(save_path)
    with open(save_path, "w") as f:
        json.dump(input, f, indent=indent)


def dict2argparse(
    input: Dict[str, Any], args: argparse.Namespace
) -> argparse.Namespace:
    for k in input.keys():
        if hasattr(args, k):
            setattr(args, k, input[k])
        else:
            logger.debug(
                f"Cannot load the value in {k} key into argparse."
                f"argparse does not contain {k} key."
            )
    return args


def json2argparse(
    json_file: str, args: argparse.Namespace
) -> argparse.Namespace:
    """Load parameters from json to `argparse`.
    Paramters from json must have same key as `argparse` to load.
    Example:
    >>> json2argparse("test.json")
    """
    assert isinstance(json_file, str)
    json_file = os.path.expanduser(json_file)
    data = load_json(json_file)
    args = dict2argparse(data, args)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing argparse to json.")
    parser.add_argument("--a", type=int, default=123)
    parser.add_argument("--b", type=int, default=456)
    args = parser.parse_args()

    dict_ = {"a": 1, "b": 4}
    dump_json(dict_, "./test.json")
    args = json2argparse(args, "./test.json")
