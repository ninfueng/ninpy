#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
import random
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from .common import multilv_getattr

def topk_accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
        k: int = 1
        ) -> torch.Tensor:
    r"""Get top-k corrected predictions and batch size.
    >>> topk_accuracy(
        torch.ones(batch_size, num_classes),
        torch.ones(batch_size))
    """
    assert isinstance(k, int)
    assert k >= 1
    pred, target = pred.detach().data, target.detach().data
    batch_size = target.shape[0]
    _, pred = pred.topk(k, dim=-1)
    # Make targets shape same as the topk pred.
    target = target.expand_as(pred.T)
    correct = target.T.eq(pred)
    return correct.numpy().sum(), batch_size


def seed_torch(seed: int = 2021, verbose: bool = True) -> None:
    r"""Seed the random seed to all possible modules.
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
            f'Save model@ {save_dir} with {epoch} epoch.')


def load_model(
    save_dir: str, model: nn.Module,
    optim = None, verbose: bool = True):
    r"""Load model from `save_dir` and extract compressed information.
    """
    assert isinstance(save_dir, str)
    ckpt = torch.load(save_dir)
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    metric, epoch = ckpt['metric'], ckpt['epoch']
    model = model.load_state_dict(model_state_dict)

    if optim is not None:
        optim = optim.load_state_dict(optimizer_state_dict)

    if verbose:
        logging.info(f'Load a model with score {metric}@ {epoch} epoch')
    return model, optim


def get_bn_names(module) -> List[str]:
    r"""Designed for using with `add_weight_decay` as `skip_list`.
    """
    name_bn_modules = []
    for n, m in module.named_modules():
        if isinstance(m, _BatchNorm):
            name_bn_modules.append(n + '.bias')
            name_bn_modules.append(n + '.weight')
    return name_bn_modules


def add_weight_decay(
    model: nn.Module,
    weight_decay: float,
    skip_list = (),
    verbose: bool = True) -> None:
    r"""Adding weight decay by avoiding batch norm and all bias.
    From:
        https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/3
        https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
        https://github.com/pytorch/pytorch/issues/1402
    Example:
    >>> add_weight_decay(model, 4e-5, (''))
    """
    assert isinstance(weight_decay, float)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Skip frozen weights.
            continue
        if(
            len(param.shape) == 1 or
            name.endswith('.bias') or
            name in skip_list):

            no_decay.append(param)
            if verbose:
                logging.info(
                    f'Skipping the weight decay on: {name}.')
        else:
            decay.append(param)

    assert len(list(model.parameters())) == len(decay) + len(no_decay)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}]


def set_warmup_lr(
    init_lr: float, warmup_epochs: int, train_loader,
    optimizer, batch_idx: int, epoch_idx: int, verbose: bool = True) -> None:
    r"""Calculate and set the warmup learning rate.
    >>> for w in range(warmup_epochs):
    >>>     for idx, (data, target) in enumerate(train_loader):
    >>>         set_warmup_lr(
                    initial_lr, warmup_epochs, train_loader,
                    optimizer, idx, w, False)
    """
    assert isinstance(warmup_epochs, int)
    total = warmup_epochs*(len(train_loader))
    iteration = (batch_idx + 1) + (epoch_idx*len(train_loader))
    lr = init_lr*(iteration/total)
    optimizer.param_groups[0]['lr'] = lr

    if verbose:
        logging.info(f'Learning rate: {lr}, Step: {iteration}/{total}')


def make_onehot(input, num_classes: int):
    r"""Convert class index tensor to one hot encoding tensor.
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


def set_batchnorm_eval(m) -> None:
    r"""From: https://discuss.pytorch.org/t/cannot-freeze-batch-normalization-parameters/38696
    Ex:
    >>> model.apply(set_batchnorm_eval)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def freeze_batchnorm(m) -> None:
    r"""
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
    r"""From: https://github.com/pytorch/examples/blob/master/dcgan/main.py
    >>> model.apply(normal_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def get_num_weight_from_name(
        model: nn.Module, name: str, verbose: bool = True) -> list:
    r"""Get a number of weight from a name of module.
    >>> model = resnet18(pretrained=False)
    >>> num_weight = get_num_weight_from_name(model, 'fc')
    """
    assert isinstance(name, str)
    module = multilv_getattr(model, name)
    num_weights = module.weight.numel()
    if verbose:
        logging.info(
            f'Module: {name} contains {num_weights} parameters.')
    return num_weights


class EarlyStoppingException(Exception):
    r"""Exception for catching early stopping. For exiting out of loop."""
    pass


class CheckPointer(object):
    r"""TODO: Adding with optimizer, model save, and unittest.
    """
    def __init__(self, task: str = 'max', patience: int = 10, verbose: bool = True) -> None:
        assert isinstance(verbose, bool)
        if task == 'max':
            self.var = np.finfo(float).min
        elif task.lower() == 'min':
            self.var = np.finfo(float).max
        else:
            raise NotImplementedError(
                f'var can be only `max` or `min`. Your {verbose}')
        self.task = task.lower()
        self.verbose = verbose
        self.patience = patience
        self.patience_counter = 0

    def update_model(self, model: nn.Module, score: float) -> None:
        r"""Save model if score is better than var.
        Raise:
            EarlyStoppingException: if `score` is not better than `var` for `patience` times.
        """
        if self.task == 'max':
            if score > self.var:
                # TODO: model saves
                model.save_state_dict()
                if self.verbose:
                    logging.info(f'Save model@{score}.')
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        elif self.task == 'min':
            if score < self.var:
                # TODO: model save
                model.save_state_dict()
                if self.verbose:
                    logging.info('Save model@{score}.')
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        if self.patience == self.patience_counter:
            raise EarlyStoppingException(
                f'Exiting: patience_counter == {self.patience}.')

        def __str__(self) -> str:
            # TODO: print and testing for which one is better str or repr.
            return (
                f'Task: {self.task} \n Best value: {self.var}\n'
                f'Counter: {self.patience_counter}\n')


if __name__ == '__main__':
    pass
