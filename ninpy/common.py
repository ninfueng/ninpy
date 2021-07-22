#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: Ninnart Fuengfusin"""
import argparse
import pickle
from typing import Any


def str2bool(v: str) -> bool:
    """From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Put str2bool as a type of argparse to able to receive true or false as input.
    """
    if isinstance(v, bool):
        return v
    lower = v.lower()
    if lower in ("yes", "true", "t", "y", "1"):
        return True
    elif lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value is expected. Your input: {v}"
        )


def multilv_getattr(obj, multi_lv: str):
    """Get multi-levels attribute.
    Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multilv_getattr(model, 'trunk.early')
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split(".")
    for l in lvs:
        obj = getattr(obj, l)
    return obj


def multilv_setattr(obj, multi_lv: str, set_with: object) -> None:
    """Set multi-levels attribute.
    Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multilv_setattr(model, 'conv_up3.conv', nn.Conv2d(5, 5, 5))
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split(".")
    for l in lvs[:-1]:
        obj = getattr(obj, l)
    setattr(obj, lvs[-1], set_with)


def pickle_dump(name: str, obj: Any) -> None:
    with open(name, "wb") as p:
        pickle.dump(obj, p, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(name: str) -> Any:
    with open(name, "rb") as f:
        return pickle.load(f)


class RunningAverage(object):
    """A simple class that maintains the running average of a quantity
    Example:
    >>> loss_avg = RunningAverage()
    >>> loss_avg.update(loss, batch_size)
    >>> loss_avg()
    """

    def __init__(self):
        self.numel = 0
        self.total = 0
        self.steps = 0

    def update(self, val: int, numel: int):
        self.total += val
        self.numel += numel
        self.steps += 1

    def __call__(self):
        return self.total / self.numel


def assertall(obj, func, *attrs):
    """Created for `hasattr`, and etc.
    Not work with func with two arguments upper.
    """
    assert callable(func)
    return all([func(obj, attr) for attr in attrs])
