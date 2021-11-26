#!/usr/bin/env python3
"""im2col related functions.
"""
import numpy as np  # type: ignore
import torch
import torch.nn as nn
from torch import Tensor


def reshape_im2col_weight(conv2d: nn.Conv2d) -> Tensor:
    """Reshape weights from nn.Conv2d to im2col format."""
    weight = conv2d.weight.reshape(
        conv2d.out_channels, conv2d.in_channels * np.prod(conv2d.kernel_size)
    )
    return weight


def reshape_im2col_activation(conv2d: nn.Conv2d, activation: Tensor) -> Tensor:
    """Reshape im2col resultant activation back to original shape."""
    kernel_size = conv2d.kernel_size
    assert kernel_size[0] == kernel_size[1]
    reshaped = activation.reshape(
        activation.shape[0],
        conv2d.out_channels,
        int(activation.shape[-1] ** 0.5),
        int(activation.shape[-1] ** 0.5),
    )
    return reshaped


def set_unfold(conv2d: nn.Conv2d) -> nn.Unfold:
    """Set unfold parameters from nn.Conv2d."""
    unfold = nn.Unfold(
        conv2d.kernel_size, conv2d.dilation, conv2d.padding, conv2d.stride
    )
    return unfold
