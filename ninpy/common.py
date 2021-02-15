#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common functions created with using only standard libraries.
@author: Ninnart Fuengfusin
"""
import time
import logging
import argparse


def str2bool(v: str) -> bool:
    """From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Put str2bool into type of argparse.
    """
    if isinstance(v, bool):
        return v

    lower = v.lower()
    if lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f'Boolean value is expected. Your input: {v}')


def timewrap(func):
    """Wrapper of function to printing out the running of the wrapped function.
    """
    def wrapped(*args, **kwargs):
        t1 = time.time()
        func = func(*args, **kwargs)
        t2 = time.time() - t1
        logging.info(
            f'Wrapper of {func.__name__}: Run with timer: \
            {time.strftime("%H:%M:%S", time.gmtime(t2))}')
        return func
    return wrapped


def multilv_getattr(obj, multi_lv: str):
    """Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multilv_getattr(model, 'trunk.early')
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split('.')
    for l in lvs:
        obj = getattr(obj, l)
    return obj


def multilv_setattr(obj, multi_lv: str, set_with: object):
    """Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multilv_setattr(model, 'conv_up3.conv', nn.Conv2d(5, 5, 5))
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split('.')
    for l in lvs[:-1]:
        obj = getattr(obj, l)
    obj = setattr(obj, lvs[-1], set_with)
