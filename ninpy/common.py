#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common functions created with using only standard libraries.
@author: Ninnart Fuengfusin
"""
import time
import logging
import argparse

import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def str2bool(v: str) -> bool:
    r"""From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Put str2bool as a type of argparse to able to receive true or false as input.
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
    r"""Wrapper of function to printing out the running of the wrapped function.
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
    r"""Get multi-levels attribute.
    Example:
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
    r"""Set multi-levels attribute.
    Example:
    >>> from fastseg import MobileV3Large
    >>> model = MobileV3Large.from_pretrained()
    >>> multilv_setattr(model, 'conv_up3.conv', nn.Conv2d(5, 5, 5))
    """
    assert isinstance(multi_lv, str)
    lvs = multi_lv.split('.')
    for l in lvs[:-1]:
        obj = getattr(obj, l)
    obj = setattr(obj, lvs[-1], set_with)


class RunningAverage(object):
    r"""From: https://github.com/cs230-stanford/cs230-code-examples
    A simple class that maintains the running average of a quantity
    Example:
    ```
    >>> loss_avg = RunningAverage()
    >>> loss_avg.update(loss, batch_size)
    >>> loss_avg()
    ```
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
        return self.total/self.numel


def show_img_torch(x: torch.Tensor, denormalize: bool = False) -> None:
    r"""Show an image from torch format.
    """
    assert isinstance(denormalize, bool)
    assert len(x.shape) == 3
    if denormalize:
        # From: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/4
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        x = inv_normalize(x)
    x = x.transpose(0, 2).detach().cpu().numpy()
    plt.imshow(x)
    plt.show()