#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:24:28 2021
@author: Ninnart Fuengfusin
"""
import yaml
from collections import OrderedDict
from data import AttributeOrderedDict


def load_yaml(yaml_file: str, with_attribute: bool = False) -> dict:
    """Refer: https://stackabuse.com/reading-and-writing-yaml-to-a-file-in-python/
    Example:
    ```
    >>> load_yaml('./config.yaml')
    ```
    """
    assert isinstance(yaml_file, str)
    with open(yaml_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if with_attribute:
        # TODO: recursive find dict and convert it into Attribute type.
        data = AttributeOrderedDict(data)
    return data


def dump_yaml(input: dict, save_loc: str) -> None:
    """Refer: https://stackabuse.com/reading-and-writing-yaml-to-a-file-in-python/
    Designed to dump hyper parameters or results as yaml file.
    TODO: update to support automatically convert object type to python type.
    Example:
    ```
    >>> input = load_yaml('./config.yaml')
    >>> dump_yaml(input, './config2.yaml')
    ```
    """
    assert isinstance(save_loc, str)
    assert isinstance(input, dict)
    with open(save_loc, 'w') as f:
        yaml.dump(input, f)


def dict2str(input: dict) -> str:
    """Given a dict recursively includes all parameters into a string.
    `::` subdict, `:` dict, and - next var.
    Example:
    ```
    v = dict2str({1:2, 3:4, 5:{6:7}, 8:{9:{10:11}}})
    > '1:2-3:4-5::6:7-8::9::10:11-'
    ```
    Args:
        input (dict): dict to reduce to a string.
    Return:
        string (str): compressed string.
    """
    string = ''
    for k, v in sorted(input.items()):
        if not isinstance(v, dict):
            string += f'{k}:{v}-'
        else:
            string += f'{k}::'
            string += dict2str(v)
    return string


def args2str(args) -> str:
    """Generate a string from argparse parser.
    """
    #args = parser.parse_args()
    args_dict = vars(args)
    string = dict2str(args_dict)
    return string


def args2yaml(args, yaml_name: str) -> None:
    args = vars(args)
    dump_yaml(args, yaml_name)


if __name__ == '__main__':
    import argparse

    test_dict = {1:2, 3:4, 5:{6:7}, 8:{9:{10:11}}}
    LOAD_CONFIG = 'test_dumped.yaml'

    dump_yaml(test_dict, LOAD_CONFIG)
    loaded_dict = load_yaml(LOAD_CONFIG)
    assert test_dict == loaded_dict
    print(dict2str(test_dict))

    parser = argparse.ArgumentParser(description='Test config.')
    parser.add_argument('--var0', type=int, default=1)
    parser.add_argument('--var1', type=int, default=2)
    args = parser.parse_args()

    args_str = args2str(args)
    print(args_str)
    #assert args_str ==''
