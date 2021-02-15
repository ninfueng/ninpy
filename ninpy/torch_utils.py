#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
import random
import logging

import torch
import numpy as np


def seed_torch(seed: int = 2020, verbose: bool = True) -> None:
    r"""Seed random seed to all possible.
    From: https://github.com/pytorch/pytorch/issues/11278
    From: https://pytorch.org/docs/stable/notes/randomness.html
    From: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    >>> seed_torch(2021)
    """
    assert isinstance(seed, int)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if verbose:
        logging.info(f'Plant a random seed: {seed}.')


def save_model(
    save_dir: str, model, optimizer,
    metric: float = None, epoch: int = None,
    verbose: bool = True) -> None:

    if model is not None:
        model = model.state_dict()
    if optimizer is not None:
        optimizer = optimizer.state_dict()

    torch.save({
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'metric': metric,
        'epoch': epoch
        }, save_dir)

    if verbose:
        logging.info(
            f'Save model@ {save_dir}'
            f'with {epoch} epoch.')


def load_model(save_dir: str, model, optim = None, verbose: bool = True):
    ckpt = torch.load(save_dir)
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    metric = ckpt['metric']
    epoch = ckpt['epoch']
    model = model.load_state_dict(model_state_dict)
    if optim is not None:
        optim = optim.load_state_dict(optimizer_state_dict)
    if verbose:
        logging.info(f'Load a model with score {metric}@ {epoch} epoch')
    return model, optim


def add_weight_decay(model, weight_decay, skip_list=()) -> None:
    """From: https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/3
    https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
    Example:
    >>> add_weight_decay(model, 4e-5, (''))
    """
    assert isinstance(weight_decay, float)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Skip frozen weights.
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def make_onehot(input, num_classes: int):
    """Convert class index tensor to one hot encoding tensor.
    Args:
        input: A tensor of shape [N, 1, *]
        num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    assert isinstance(num_classes, int)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).to(input.device)
    result = result.scatter_(1, input, 1)
    return result


class RunningAverage(object):
    """From: https://github.com/cs230-stanford/cs230-code-examples
    A simple class that maintains the running average of a quantity
    Example:
    ```
    >>> loss_avg = RunningAverage()
    >>> loss_avg.update(2)
    >>> loss_avg.update(4)
    >>> loss_avg() = 3
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


def set_batchnorm_eval(m) -> None:
    """From: https://discuss.pytorch.org/t/cannot-freeze-batch-normalization-parameters/38696
    Ex:
    >>> model.apply(set_batchnorm_eval)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def freeze_batchnorm(m) -> None:
    """
    Ex:
    >>> model.apply(freeze_batchnorm)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for param in m.parameters():
            param.requires_grad = False


def freeze_param_given_name(
    m, freeze_names: list, verbose: bool = True) -> None:
    for name, param in m.named_parameters():
        if name in freeze_names:
            param.requires_grad = False
            if verbose:
                logging.info(
                    f'Layer: {name} was freeze.')


def normal_init(m):
    """From: https://github.com/pytorch/examples/blob/master/dcgan/main.py
    >>> model.apply(normal_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
