import torch
import torch.nn as nn

from typing import Tuple


def precompute_batchnorm(
    layer: nn.BatchNorm2d,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute BatchNorm into from of `weight`*x + `bias`. This reduces
    overall computation time.
    """
    with torch.no_grad():
        mean, var = layer.running_mean, layer.running_var
        gamma, beta, eps = layer.weight, layer.bias, layer.eps
        denominator = (var + eps).sqrt()
        weight = gamma / denominator
        bias = (-(mean * gamma) / denominator) + beta
    return weight, bias
