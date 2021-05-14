#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: Ninnart Fuengfusin"""
import argparse
import logging
import os
import random
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import hparams

from .common import multilv_getattr
from .data import AttributeOrderedDict
from .experiment import set_experiment
from .log import set_logger
from .yaml2 import load_yaml, name_experiment

warnings.warn("torch_utils is duplicated with torch2.")

def torch2numpy(x: torch.Tensor) -> np.ndarray:
    r"""Converting torch format tensor to `numpy` or `tensorflow` format."""
    assert isinstance(x, torch.Tensor)
    x = x.detach().cpu().numpy()
    if len(x.shape) == 2:
        x = np.transpose(x, (1, 0))
    elif len(x.shape) == 3:
        x = np.transpose(x, (1, 2, 0))
    elif len(x.shape) == 4:
        x = np.transpose(x, (2, 3, 1, 0))
    else:
        raise ValueError(
            f"Not supporting with shape of {len(x.shape)}, please update this function to support it."
        )
    return x


def tensorboard_models(
    writer: SummaryWriter, model: nn.Module, idx: int
) -> SummaryWriter:
    """Tracking all parameters with Tensorboard."""
    assert isinstance(idx, int)
    for name, param in model.named_parameters():
        writer.add_histogram(os.path.join("parameters", name), param, idx)
    return writer


def tensorboard_hparams(writer, hparam_dict, metric_dict) -> None:
    """Modified: https://github.com/lanpa/tensorboardX/issues/479"""
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(k, v)


def topk_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> Tuple[torch.Tensor, int]:
    """Get top-k corrected predictions and batch size.

    Example:
    >>> output = torch.tensor([[0.0, 0.1, 0.3, 0.9], [0.0, 0.8, 0.3, 0.4]])
    >>> target = torch.tensor([3, 1])
    >>> acc = accuracy(output, target)
    ([tensor(2.)], 2)
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k)
    return res, batch_size


def seed_torch(seed: int = 2021, benchmark: bool = False, verbose: bool = True) -> None:
    """Seed the random seed to all possible modules.

    From: https://github.com/pytorch/pytorch/issues/11278
        https://pytorch.org/docs/stable/notes/randomness.html
    Example:
    >>> seed_torch(2021)
    """
    assert isinstance(seed, int)
    assert isinstance(benchmark, bool)
    assert isinstance(verbose, bool)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if benchmark:
        # There is some optimized algorithm in case fixed size data.
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
    if verbose:
        logging.info(f"Plant a random seed: {seed} with benchmark mode: {benchmark}.")


def speed_torch():
    """For speed training without using any random seeds.

    Example:
    >>> speed_torch()
    """
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ninpy_setting(
    name_parser: str,
    yaml_file: Optional[str] = None,
    exp_pth: Optional[str] = None,
    to_console: bool = False,
    benchmark: bool = False,
    verbose: bool = True,
) -> Tuple[dict, str, Callable]:
    r"""Basic setting to utilize all features from ninpy.
    Get args, path to experiment folder, and, SummaryWriter.
    """
    assert isinstance(name_parser, str)
    parser = argparse.ArgumentParser(description=name_parser)
    parser.add_argument("--yaml", type=str, default=yaml_file)
    parser.add_argument("--exp_pth", type=str, default=exp_pth)
    args = parser.parse_args()

    hparams = AttributeOrderedDict(load_yaml(args.yaml))
    assert hasattr(hparams, "seed"), "yaml file should contain a seed attribute."

    if args.exp_pth == None:
        exp_pth = name_experiment(hparams)
    else:
        exp_pth = args.exp_pth

    set_experiment(exp_pth)
    set_logger(os.path.join(exp_pth, "info.log"), to_console)
    seed_torch(hparams.seed, benchmark=benchmark)
    writer = SummaryWriter(exp_pth)
    if verbose:
        logging.info(f"Ninpy settings: hparams {hparams}@ {exp_pth}")
    return hparams, exp_pth, writer


def save_model(
    save_dir: str,
    model,
    optimizer,
    amp=None,
    metric: float = None,
    epoch: int = None,
    save_epoch: int = None,
    rm_old: bool = True,
    verbose: bool = True,
) -> None:
    r"""Save model, optimizer, amp, best_metric, best_epoch into a ckpt.
    Can automatically remove old save ckpt.
    """
    assert isinstance(save_dir, str)
    assert (
        save_dir.find(".pth") > -1
    ), "Should contains type as `.pth`, otherwise not support removing old files."

    def _save_model(
        save_dir,
        model_name,
        optimizer_name,
        model,
        optimizer,
        amp,
        metric,
        epoch,
        rm_old,
        verbose,
    ):
        if rm_old:
            save_pth = os.path.dirname(save_dir)
            rm_list = [
                os.path.join(save_pth, i)
                for i in os.listdir(save_pth)
                if i.find(".pth") > -1
            ]
            [os.remove(r) for r in rm_list]

        # Still in _save_model.
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model,
                "optimizer_name": optimizer_name,
                "optimizer_state_dict": optimizer,
                "metric": metric,
                "epoch": epoch,
                "amp_state_dict": amp,
            },
            save_dir,
        )

        if verbose:
            logging.info(f"Save model@ {save_dir} with {epoch} epoch.")

    model = model.state_dict()
    model_name = model.__class__.__name__
    optimizer = optimizer.state_dict()
    optimizer_name = optimizer.__class__.__name__

    if amp is not None:
        amp = amp.state_dict()

    if save_epoch is not None:
        if epoch >= save_epoch:
            _save_model(
                save_dir,
                model_name,
                optimizer_name,
                model,
                optimizer,
                amp,
                metric,
                epoch,
                rm_old,
                verbose,
            )
    else:
        _save_model(
            save_dir,
            model_name,
            optimizer_name,
            model,
            optimizer,
            amp,
            metric,
            epoch,
            rm_old,
            verbose,
        )


def load_model(
    save_dir: str, model: nn.Module, optimizer=None, amp=None, verbose: bool = True
):
    r"""Load model from `save_dir` and extract compressed information."""
    assert isinstance(save_dir, str)
    ckpt = torch.load(save_dir)
    model_state_dict = ckpt["model_state_dict"]
    optimizer_state_dict = ckpt["optimizer_state_dict"]
    amp_state_dict = ckpt["amp_state_dict"]
    model_name = ckpt["model_name"]
    optimizer_name = ckpt["optimizer_name"]
    metric, epoch = ckpt["metric"], ckpt["epoch"]
    model.load_state_dict(model_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if amp is not None:
        amp.load_state_dict(amp_state_dict)
    if verbose:
        logging.info(
            f"Load a model {model_name} and an optimizer {optimizer_name} with score {metric}@ {epoch} epoch"
        )
    return model, optimizer


def add_weight_decay(
    model: nn.Module, weight_decay: float, skip_list=(), verbose: bool = True
) -> None:
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
            # Skip frozen weights or not require grad variables.
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
            if verbose:
                logging.info(f"Skipping the weight decay on: {name}.")
        else:
            decay.append(param)

    assert len(list(model.parameters())) == len(decay) + len(no_decay)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def set_warmup_lr(
    init_lr: float,
    warmup_epochs: int,
    train_loader,
    optimizer,
    batch_idx: int,
    epoch_idx: int,
    verbose: bool = True,
) -> None:
    r"""Calculate and set the warmup learning rate.
    >>> for w in range(warmup_epochs):
    >>>     for idx, (data, target) in enumerate(train_loader):
    >>>         set_warmup_lr(
                    initial_lr, warmup_epochs, train_loader,
                    optimizer, idx, w, False)
    """
    assert isinstance(warmup_epochs, int)
    total = warmup_epochs * (len(train_loader))
    iteration = (batch_idx + 1) + (epoch_idx * len(train_loader))
    lr = init_lr * (iteration / total)
    optimizer.param_groups[0]["lr"] = lr

    if verbose:
        logging.info(f"Learning rate: {lr}, Step: {iteration}/{total}")


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


def get_bn_names(module: nn.Module) -> List[str]:
    r"""Designed for using with `add_weight_decay` as `skip_list`."""
    name_bn_modules = []
    for n, m in module.named_modules():
        if isinstance(m, _BatchNorm):
            name_bn_modules.append(n + ".bias")
            name_bn_modules.append(n + ".weight")
    return name_bn_modules


def set_batchnorm_eval(m) -> None:
    r"""From: https://discuss.pytorch.org/t/cannot-freeze-batch-normalization-parameters/38696
    Ex:
    >>> model.apply(set_batchnorm_eval)
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def freeze_batchnorm(m) -> None:
    r"""
    Ex:
    >>> model.apply(freeze_batchnorm)
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        for param in m.parameters():
            param.requires_grad = False


def freeze_param_given_name(m, freeze_names: list, verbose: bool = True) -> None:
    for name, param in m.named_parameters():
        if name in freeze_names:
            param.requires_grad = False

            if verbose:
                logging.info(f"Layer: {name} was freeze.")


def normal_init(m):
    r"""From: https://github.com/pytorch/examples/blob/master/dcgan/main.py
    >>> model.apply(normal_init)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def get_num_weight_from_name(model: nn.Module, name: str, verbose: bool = True) -> list:
    r"""Get a number of weight from a name of module.
    >>> model = resnet18(pretrained=False)
    >>> num_weight = get_num_weight_from_name(model, 'fc')
    """
    assert isinstance(name, str)
    module = multilv_getattr(model, name)
    num_weights = module.weight.numel()
    if verbose:
        logging.info(f"Module: {name} contains {num_weights} parameters.")
    return num_weights


class EarlyStoppingException(Exception):
    r"""Exception for catching early stopping. For exiting out of loop."""
    pass


class CheckPointer(object):
    r"""TODO: Adding with optimizer, model save, and unittest."""

    def __init__(
        self, task: str = "max", patience: int = 10, verbose: bool = True
    ) -> None:
        assert isinstance(verbose, bool)
        if task == "max":
            self.var = np.finfo(float).min
        elif task.lower() == "min":
            self.var = np.finfo(float).max
        else:
            raise NotImplementedError(f"var can be only `max` or `min`. Your {verbose}")
        self.task = task.lower()
        self.verbose = verbose
        self.patience = patience
        self.patience_counter = 0

    def update_model(self, model: nn.Module, score: float) -> None:
        r"""Save model if score is better than var.
        Raise:
            EarlyStoppingException: if `score` is not better than `var` for `patience` times.
        """
        if self.task == "max":
            if score > self.var:
                # TODO: model saves
                model.save_state_dict()
                if self.verbose:
                    logging.info(f"Save model@{score}.")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        elif self.task == "min":
            if score < self.var:
                # TODO: model save
                model.save_state_dict()
                if self.verbose:
                    logging.info("Save model@{score}.")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        if self.patience == self.patience_counter:
            raise EarlyStoppingException(
                f"Exiting: patience_counter == {self.patience}."
            )

        def __str__(self) -> str:
            # TODO: print and testing for which one is better str or repr.
            return (
                f"Task: {self.task} \n Best value: {self.var}\n"
                f"Counter: {self.patience_counter}\n"
            )


class SummaryWriterDictList(SummaryWriter):
    r"""SummaryWriter with support the adding multiple scalers to dictlist."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def add_scalar_from_dict(self, counter: int = None, kwargs=None) -> None:
        """Broadcast add_scalar to all elements in dict."""
        for key in kwargs.keys():
            if counter is not None:
                self.add_scalar(str(key), kwargs[key], counter)
            else:
                self.add_scalar(str(key), kwargs[key])

    def add_scalar_from_kwargs(self, counter: int = None, **kwargs) -> None:
        """Broadcast add_scalar to all elements in dict."""
        for key in kwargs.keys():
            if counter is not None:
                self.add_scalar(str(key), kwargs[key], counter)
            else:
                self.add_scalar(str(key), kwargs[key])
