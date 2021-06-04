#!/usr/bin/env python3
"""Base datasets"""
import glob
import os
from functools import reduce
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader

from ninpy.datasets.utils import IMG_EXTENSIONS, multithread_load_images

__all__ = ["BaseDataset", "BurstDataset", "BurstImageFolder"]


class BaseDataset(Dataset):
    """BaseDataset for using as a template for other Dataset.
    Arguments:
        loader (Callable): a function to load images.
        target_loader (Callable): a function to load labels and preprocess in
        a preferable format.
    """

    def __init__(
        self,
        root: str,
        loader: Callable,
        target_loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        assert os.path.isdir(self.root)

        self.transform = (lambda x: x) if transform is None else transform
        self.target_transform = (
            (lambda x: x) if target_transform is None else target_transform
        )
        self.loader = (lambda x: x) if loader is None else loader
        self.target_loader = (lambda x: x) if target_loader is None else target_loader
        self.data_dirs, self.labels = self.get_data_dirs_labels()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data, label = self.data_dirs[idx], self.labels[idx]
        data, label = self.loader(data), self.target_loader(label)
        data, label = self.transform(data), self.target_transform(label)
        return data, label

    def __len__(self) -> int:
        return len(self.data_dirs)

    def get_data_dirs_labels(self) -> None:
        """Get all data locations and labels. Can inputs more data to further processing."""
        raise NotImplementedError()


class BurstDataset(BaseDataset):
    """Loading all images and labels to RAM for faster access.
    >>> dataset = BurstDataset()
    >>> dataset.load_images()
    >>> dataset.load_labels()
    """

    def __init__(
        self,
        root: str,
        loader: Callable,
        target_loader: Callable = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, loader, target_loader, transform, target_transform)

    def load_images(
        self, load_images_fn: Callable, num_workers: int = cpu_count()
    ) -> None:
        """Load images and assign identify function to self.loader."""
        images = multithread_load_images(self.data_dirs, load_images_fn, num_workers)
        self.data_dirs = images
        self.loader = lambda x: x

    def load_labels(self) -> None:
        """Load labels and preprocess labels, Ex: prepare labels for object detection."""
        # TODO: support multi-thread load and process labels?
        self.labels = [self.target_loader(i) for i in self.labels]
        self.target_loader = lambda x: x


class BurstImageFolder(ImageFolder):
    """ImageFolder instead of load images at a runtime.
    Load all images to a RAM to faster access.
    Example:
    >>> traindir = os.path.expanduser("~/datasets/CINIC10/train")
    >>> dataset = BurstImageFolder(traindir)
    >>> dataset.load_images()
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable = pil_loader,
        is_valid_file: Optional[bool] = None,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.classes = sorted(os.listdir(self.root))
        transform = (lambda x: x) if transform is None else transform
        target_transform = (
            (lambda x: x) if target_transform is None else target_transform
        )
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.samples = self.load_images()
        self.loader = lambda x: x

    def load_images(self) -> None:
        img_dirs = [self.get_img_dirs(os.path.join(self.root, c)) for c in self.classes]
        num_labels = [len(i) for i in img_dirs]
        labels = []
        for idx, n in enumerate(num_labels):
            for _ in range(n):
                labels.append(idx)
        img_dirs = reduce(lambda x, y: x + y, img_dirs)
        imgs = multithread_load_images(img_dirs)
        return [(i, d) for i, d in zip(imgs, labels)]

    def get_img_dirs(self, path: str) -> List[str]:
        assert isinstance(path, str)
        img_dirs = []
        for e in IMG_EXTENSIONS:
            img_dirs += glob.glob(os.path.join(path, "*" + e))
            img_dirs += glob.glob(os.path.join(path, "*" + e.upper()))
        return img_dirs


if __name__ == "__main__":
    dataset = BurstImageFolder("~/datasets/CINIC10/train")
    dataset.load_images()
    img, label = next(iter(dataset))
    assert img.size == (32, 32)
