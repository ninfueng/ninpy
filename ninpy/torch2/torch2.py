#!/usr/bin/env python3
"""@author: Ninnart Fuengfusin"""
import argparse
import logging
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.conv import _ConvNd
from torch.optim.lr_scheduler import _LRScheduler

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

from ninpy.common import multilv_getattr
from ninpy.data import AttrDict
from ninpy.experiment import set_experiment
from ninpy.log import set_logger
from ninpy.yaml2 import load_yaml, name_experiment


def torch2np(x: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor format (NCHW) to Numpy or TensorFlow format (NHWC)."""
    assert isinstance(x, torch.Tensor)
    x = x.detach().cpu()
    shape = x.shape
    if len(shape) == 2:
        x = torch.movedim(x, 1, 0)
    elif len(shape) == 3:
        x = torch.movedim(x, 0, 2)
    elif len(shape) == 4:
        x = x.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Not supporting with shape of {len(shape)}.")
    x = x.numpy()
    return x


def np2torch(x: np.ndarray) -> torch.Tensor:
    """Convert Numpy tensor format (NHWC) to PyTorch format (NCHW)."""
    shape = x.shape
    x = torch.as_tensor(x)
    if len(shape) == 2:
        x = torch.movedim(x, -1, 0)
    elif len(shape) == 3:
        x = torch.movedim(x, -1, 0)
    elif len(shape) == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Not supporting with shape of {len(shape)}.")
    return x


def get_mean_std(dataset, burst: bool = True) -> Tuple[float, float]:
    """Get a mean and standard deviation from a dataset.
    Args:
        burst (bool): If True load all data to RAM and calculates a mean and standard deviation.
        Else accumulate all information into a
    """
    sample, _ = next(iter(dataset))
    assert (
        True if isinstance(sample, torch.Tensor) else False
    ), "Support only PyTorch format."
    if burst:
        dataset = torch.stack([i[0] for i in list(dataset)], dim=0)
        mean = torch.mean(dataset, dim=(0, 2, 3))
        std = torch.std(dataset, dim=(0, 2, 3))
    else:
        # TODO: Fix this incremental of std.
        raise NotImplementedError("Currently support only burst=True.")
        # size = len(dataset)
        # mean = std = 0.0
        # for d, _ in dataset:
        #     mean += d.mean(dim=(1, 2))
        # mean /= size
        # for d, _ in dataset:
        #     std += (d.mean(dim=(1, 2)) - mean).pow(2)
        # std /= size
        # std = std.sqrt()
    return mean, std


def set_fastmode():
    """
    Refer:
        https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
        https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD
    """
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)


def tensorboard_models(
    writer: SummaryWriter, model: nn.Module, idx: int
) -> SummaryWriter:
    """Tracking all parameters with Tensorboard."""
    assert isinstance(idx, int)
    for name, param in model.named_parameters():
        writer.add_histogram(os.path.join("parameters", name), param, idx)
    return writer


def tensorboard_hparams(
    writer: SummaryWriter, hparam_dict: dict, metric_dict: dict
) -> None:
    """Modified: https://github.com/lanpa/tensorboardX/issues/479"""
    # In PyTorch 1.2, this `hparams` is not supported.
    # This should not affect a performance because this function should be applied only once.
    from torch.utils.tensorboard.writer import hparams

    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(os.path.join("hyperparameters", k), v)


def topk_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> Tuple[torch.Tensor, int]:
    """Modified: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Get top-k corrected predictions and batch size.

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
    corrects = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        corrects.append(correct_k)
    return corrects, batch_size


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


def ninpy_setting(
    name_parser: str,
    yaml_file: Optional[str] = None,
    exp_pth: Optional[str] = None,
    to_console: bool = False,
    benchmark: bool = False,
    verbose: bool = True,
) -> Tuple[dict, str, Callable]:
    """Basic initial setting to utilize all features from ninpy.
    Get args, path to experiment folder, and, SummaryWriter.
    """
    assert isinstance(name_parser, str)
    parser = argparse.ArgumentParser(description=name_parser)
    parser.add_argument("--yaml", type=str, default=yaml_file)
    parser.add_argument("--exp_pth", type=str, default=exp_pth)
    args = parser.parse_args()

    hparams = AttrDict(load_yaml(args.yaml))
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
        logging.info("Initial ninpy setting")
    return hparams, exp_pth, writer


def save_model(
    save_dir: str,
    model: nn.Module,
    optimizer,
    amp=None,
    metric: float = None,
    epoch: int = None,
    save_epoch: int = None,
    rm_old: bool = True,
    verbose: bool = True,
) -> None:
    r"""Save model state_dict, model name, optimizer state_dict, optimizer name,
    amp state_dict, best_metric, best_epoch into a pth. Can automatically remove old pth.
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
    save_dir: str,
    model: nn.Module,
    optimizer=None,
    amp=None,
    scheduler=None,
    verbose: bool = True,
) -> Optional[int]:
    r"""Load model from `save_dir` and extract compressed information.
    Return:
        epoch (int): a number of epoch that is already trained.
    """
    assert isinstance(save_dir, str)
    ckpt = torch.load(save_dir)

    # Checking each key is exist or not and load it.
    if "model_state_dict" in ckpt.keys():
        model_state_dict = ckpt["model_state_dict"]
        model.load_state_dict(model_state_dict)
        logging.info("Load a model_state_dict.")
    else:
        model_state_dict = None
    if "optimizer_state_dict" in ckpt.keys():
        optimizer_state_dict = ckpt["optimizer_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
        logging.info("Load an optimizer_state_dict.")
    else:
        optimizer_state_dict = None
    if "amp_state_dict" in ckpt.keys():
        amp_state_dict = ckpt["amp_state_dict"]
        amp.load_state_dict(amp_state_dict)
    else:
        amp_state_dict = None

    if "model_name" in ckpt.keys():
        model_name = ckpt["model_name"]
    else:
        model_name = None
    if "optimizer_name" in ckpt.keys():
        optimizer_name = ckpt["optimizer_name"]
    else:
        optimizer_name = None
    if "metric" in ckpt.keys():
        metric = ckpt["metric"]
    else:
        metric = None
    if "epoch" in ckpt.keys():
        epoch = ckpt["epoch"]
    else:
        epoch = None

    if scheduler is not None and epoch is not None:
        try:
            for _ in range(epoch):
                scheduler.step()
        except TypeError:
            raise TypeError("{scheduler.__class__.__name__()}: requires input metrics.")

    if verbose:
        logging.info(
            f"Load a model {model_name} and an optimizer {optimizer_name}"
            f" with score {metric}@ {epoch} epoch"
        )
    return epoch


def add_weight_decay(
    model: nn.Module, weight_decay: float
) -> List[Dict[str, torch.Tensor]]:
    """Adding weight decay by avoiding batch norm and all bias."""

    decay_group, not_decay_group, skip_group = [], [], []
    for m in model.modules():
        if isinstance(m, (nn.Linear, _ConvNd)):
            decay_group.append(m.weight)
            if m.bias is not None:
                not_decay_group.append(m.bias)
        elif isinstance(m, _NormBase):
            if m.weight is not None:
                not_decay_group.append(m.weight)
            if m.bias is not None:
                not_decay_group.append(m.bias)
        else:
            skip_group.append(m)
    assert len(list(model.parameters())) == len(decay_group) + len(
        not_decay_group
    ), "Number of detected parameters are not equal,"
    f"maybe in some of these parameters in this list: {m}"

    return [
        {"params": not_decay_group, "weight_decay": 0.0},
        {"params": decay_group, "weight_decay": weight_decay},
    ]


def init_weights(model: nn.Module) -> None:
    """Initialize weights and biases depending on the type of layer.
    https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
    https://github.com/aliyun/alibabacloud-quantization-networks/blob/master/models/alexnet_all.py
    """
    for m in model.modules():
        if isinstance(m, _ConvNd):
            # a = 0, in case of leaky relu consider change this.
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, _NormBase):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            # Same as PyTorch hyper-parameters.
            m.eps = 1e-5
            m.momentum = 0.1
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


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


def set_batchnorm_eval(m) -> None:
    r"""From: https://discuss.pytorch.org/t/cannot-freeze-batch-normalization-parameters/38696
    Ex:
    >>> model.apply(set_batchnorm_eval)
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def freeze_batchnorm(m: nn.Module) -> None:
    """>>> model.apply(freeze_batchnorm)"""
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


def get_num_weight_from_name(model: nn.Module, name: str, verbose: bool = True) -> list:
    """Get a number of weight from a name of module.
    >>> model = resnet18(pretrained=False)
    >>> num_weight = get_num_weight_from_name(model, 'fc')
    """
    assert isinstance(name, str)
    module = multilv_getattr(model, name)
    num_weights = module.weight.numel()
    if verbose:
        logging.info(f"Module: {name} contains {num_weights} parameters.")
    return num_weights


class CheckPointer(object):
    """TODO: Adding with optimizer, model save, and unittest."""

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
    """SummaryWriter with support the adding multiple scalers to dictlist."""

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


class BatchWarmupScheduler(_LRScheduler):
    """Batch-wise warmup learning rate.
    Use this after another scheduler, otherwise the learning rate may be weird.
    """

    def __init__(
        self,
        optimizer: torch.optim,
        train_loader: torch.utils.data.DataLoader,
        warmup_epochs: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.warmup_batchs = self._get_warmup_batchs(train_loader, warmup_epochs)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> None:
        assert (
            self.warmup_batchs >= self.last_epoch
        ), f"Number of step is more than number of warmup_steps: {self.warmup_steps}"
        if self.last_epoch == 0:
            # Settings for initial learning rate.
            return [0.0 for _ in self.optimizer.param_groups]
        elif self.last_epoch == 1:
            # Make sure that initial learning rate is correct with other schedulers.
            # This if cause can protect only a scheduler only!
            return [0.0 + 1 / self.warmup_steps for _ in self.optimizer.param_groups]
        # After each step adding 1/self.warmup_steps.
        return [
            1 / self.warmup_steps + group["lr"] for group in self.optimizer.param_groups
        ]

    def _get_warmup_batchs(
        self, train_loader: torch.utils.data.DataLoader, warmup_epochs: int
    ) -> int:
        warmup_batchs = len(train_loader) * warmup_epochs
        return warmup_batchs
