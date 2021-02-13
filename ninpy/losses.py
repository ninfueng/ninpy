#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:24:33 2021

@author: ninnart
"""
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def cal_cls_weights(loader, num_classes: int):
    """Given masks, finding occurances for each class.
    Make in 1/occurance and to convert into range [0, 1].
    """
    assert isinstance(num_classes, int)
    accum_count = Counter()
    for _, mask in tqdm(loader):
        mask = mask.numpy()
        uniq, count = np.unique(mask, return_counts=True)
        pair_count = {u:c for u, c in zip(uniq.tolist(), count.tolist())}
        accum_count.update(pair_count)
    assert num_classes == len(accum_count.keys())
    weights = np.array([accum_count[k] for k in range(num_classes)])
    weights = (1/weights)/((1/weights).sum())
    return weights


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing loss.
    From: https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))