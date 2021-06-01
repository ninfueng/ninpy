#!/usr/bin/env python3
"""Base datasets"""

import torch.nn as nn
from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    """TODO: thinking about the feature of BaseDataset."""
    NUM_CLASSES = None
    CLASSES = None

    def __init__(self):
        super().__init__()



