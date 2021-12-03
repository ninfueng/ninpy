#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: Ninnart Fuengfusin"""
import argparse
import datetime
from typing import Any

__all__ = [
    "get_datetime",
    "str2bool",
    "multi_getattr",
    "multi_setattr",
]


def get_datetime() -> str:
    """Return a string of year-month-day-hour-minute-second."""
    date = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    date = date.replace(":", "-").replace(".", "-").replace(" ", "-")
    return date


def str2bool(v: str) -> bool:
    """Modified from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Put str2bool as a type of argparse to able to receive true or false as input.
    Example:
        >>> parser.add_argument("--book", type=str2bool)
        python main.py --book true
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
            f"Boolean value is expected. Your input: {v}."
        )


def multi_getattr(obj: object, multi_attr: str):
    """Get multi-levels attribute.
    Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multi_getattr(model, 'trunk.early')
    """
    assert isinstance(multi_attr, str)
    attrs = multi_attr.split(".")
    for a in attrs:
        obj = getattr(obj, a)
    return obj


def multi_setattr(obj: object, multi_attr: str, value: Any) -> None:
    """Set multi-levels attribute.
    Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multi_setattr(model, 'conv_up3.conv', nn.Conv2d(5, 5, 5))
    """
    assert isinstance(multi_attr, str)
    attrs = multi_attr.split(".")
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attrs[-1], value)
