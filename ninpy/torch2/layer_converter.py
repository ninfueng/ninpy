#!/usr/bin/env python3
"""@author: Ninnart Fuengfusin."""
import logging
from typing import Callable, Optional

import torch.nn as nn


class LayerConverter(object):
    """Collection of converter for layer to another type of layer.
    Supports:
        Conv, Linear, Activation, and BatchNorm.
    Modified from:
        https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104686
    """

    @staticmethod
    def cvt_batchnorm(
        model: nn.Module,
        old_layer_type: Callable,
        new_layer_type: Callable,
        convert_weights: bool = False,
        num_groups=Optional[int],
    ) -> nn.Module:
        """If `num_groups` is 1, GroupNorm turns into LayerNorm.
        If `num_groups` is None, GroupNorm turns into InstanceNorm.
        Example:
        >>> LayerConverter.cvt_batchnorm(model, nn.BatchNorm2d, nn.GroupNorm, True, num_groups=2)
        """
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.cvt_batchnorm(
                    module,
                    old_layer_type,
                    new_layer_type,
                    convert_weights,
                    num_groups=num_groups,
                )
            # single module
            if type(module) == old_layer_type:
                old_layer = module
                new_layer = new_layer_type(
                    module.num_features if num_groups is None else num_groups,
                    module.num_features,
                    module.eps,
                    module.affine,
                )
                if convert_weights:
                    new_layer.weight = old_layer.weight
                    new_layer.bias = old_layer.bias
                model._modules[name] = new_layer
        return model

    @staticmethod
    def cvt_conv2d(
        model: nn.Module,
        old_layer_type: Callable,
        new_layer_type: Callable,
        convert_weights: bool = False,
        **kwargs,
    ) -> nn.Module:
        """Example:
        >>> from torchvision.models import vgg16
        >>> m = vgg16(pretrained=False)
        >>> LayerConverter.convert_conv2d(m, nn.Conv2d, nn.Conv1d)
        """
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.cvt_conv2d(
                    module,
                    old_layer_type,
                    new_layer_type,
                    convert_weights,
                    **kwargs,
                )
            # single module
            if type(module) == old_layer_type:
                old_layer = module
                new_layer = new_layer_type(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=not module.bias is None,
                    **kwargs,
                )
                if convert_weights:
                    new_layer.weight = old_layer.weight
                    new_layer.bias = old_layer.bias
                model._modules[name] = new_layer
        return model

    @staticmethod
    def cvt_linear(
        model,
        old_layer_type: Callable,
        new_layer_type: Callable,
        convert_weights: bool = False,
        **kwargs,
    ) -> nn.Module:
        """Example:
        >>> from torchvision.models import vgg16
        >>> m = vgg16(pretrained=False)
        >>> LayerConverter.convert_linear(m, nn.Linear, BinLinear)
        """
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.cvt_linear(
                    module,
                    old_layer_type,
                    new_layer_type,
                    convert_weights,
                    **kwargs,
                )
            # single module
            if type(module) == old_layer_type:
                old_layer = module
                new_layer = new_layer_type(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=not module.bias is None,
                    **kwargs,
                )
                if convert_weights:
                    new_layer.weight = old_layer.weight
                    new_layer.bias = old_layer.bias
                model._modules[name] = new_layer
        return model

    @staticmethod
    def cvt_activation(
        model: nn.Module,
        old_layer_type: Callable,
        new_layer_type: Callable,
        **kwargs,
    ) -> nn.Module:
        """Example:
        >>> from torchvision.models import vgg16
        >>> m = vgg16(pretrained=False)
        >>> LayerConverter.cvt_activation(m, nn.ReLU, nn.ReLU6)
        """
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.cvt_activation(
                    module, old_layer_type, new_layer_type, **kwargs
                )
            # single module
            if type(module) == old_layer_type:
                try:
                    new_layer = new_layer_type(inplace=module.inplace, **kwargs)
                except TypeError:
                    # A problem with third party built activation.
                    # Some activation may not contains the inplace attribute.
                    # If exception raised, recreates module without an inplace argument.
                    new_layer = new_layer_type(**kwargs)
                model._modules[name] = new_layer
        return model


def convert_module2module(
    model, module_a, module_b, verbose: bool = True
) -> None:
    r"""From: https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/5
    Example:
    >>> convert_module2module(model, HardSwish, nn.Hardswish())
    """
    assert isinstance(verbose, bool)
    for child_name, child in model.named_children():
        if isinstance(child, module_a):
            setattr(model, child_name, module_b)
            if verbose:
                logging.info(f"Replace {module_a} to {module_b}")
        else:
            convert_module2module(child, module_a, module_b)
