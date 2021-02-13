#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:22:20 2021

@author: ninnart
"""
import time
import logging


def time_wrap(func):
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
    """Ex:
        from fastseg import MobileV3Large 
        model = MobileV3Large.from_pretrained()
        multilv_getattr(model, 'trunk.early')
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split('.')
    for l in lvs:
        obj = getattr(obj, l)
    return obj


def multilv_setattr(obj, multi_lv: str, set_with: object):
    """Ex:
        from fastseg import MobileV3Large 
        model = MobileV3Large.from_pretrained()
        multilv_setattr(model, 'conv_up3.conv', nn.Conv2d(5, 5, 5))
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split('.')
    for l in lvs[:-1]:
        obj = getattr(obj, l)
    obj = setattr(obj, lvs[-1], set_with)

