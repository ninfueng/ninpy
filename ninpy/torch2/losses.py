#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ["get_class_weights", "LabelSmoothLoss"]


def get_class_weights(dataset: Dataset) -> np.ndarray:
    """Get masks from a Dataset, finding occurrences for each class.
    Make in 1/occurrences and to convert into range [0, 1].
    May return more classes than
    """
    accum_count = Counter()
    pbar = tqdm(dataset)
    pbar.set_description("Calculate for class weights")
    for _, mask in dataset:
        unique, count = np.unique(mask, return_counts=True)
        pair_count = {u: c for u, c in zip(unique.tolist(), count.tolist())}
        accum_count.update(pair_count)

    num_classes = len(accum_count.keys())
    weights = np.array([accum_count[k] for k in range(num_classes)])
    weights = (1 / weights) / ((1 / weights).sum())
    return weights


class LabelSmoothLoss(nn.Module):
    """Label Smoothing loss. Cross entropy loss.
    From:
        https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """

    def __init__(
        self, num_classes: int, smooth: float = 0.0, dim: int = -1
    ) -> None:
        super().__init__()
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred, target) -> torch.Tensor:
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            # One hot vector.
            true_dist.fill_(self.smooth / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
