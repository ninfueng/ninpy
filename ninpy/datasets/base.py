#!/usr/bin/env python3
"""Base datasets"""

import torch.nn as nn
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """TODO: thinking about the feature of BaseDataset."""

    NUM_CLASSES = CLASSES = None

    def __init__(self) -> None:
        super().__init__()

        self.transform = None
        self.target_transform = None
        self.img_dirs = []

    def load_img(self):
        return


class BurstDataset(BaseDataset):
    """Loading all images to RAM for faster access.
    Support with
    """

    def __init__(self) -> None:
        super().__init__()


class SegmentBaseDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.images = []


class DetectDataset(BaseDataset):
    """Loading all images to RAM for faster access."""

    def __init__(self) -> None:
        super().__init__()


class SegmentBurstDataset(BurstDataset):
    """Loading all images to RAM for faster access."""

    def __init__(self) -> None:
        super().__init__()


class DetectBurstDataset(BurstDataset):
    """Loading all images to RAM for faster access."""

    def __init__(self) -> None:
        super().__init__()
