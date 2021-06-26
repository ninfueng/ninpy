#!/usr/bin/env python3
"""Base datasets"""
import glob
import os
import warnings
from functools import reduce
from multiprocessing import cpu_count
from typing import Any, Callable, List, Optional, Tuple

import psutil
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader

from ninpy.datasets.utils import IMG_EXTENSIONS, multithread_load_images

__all__ = ["BaseDataset", "BurstDataset", "BurstImageFolder"]


class BaseDataset(Dataset):
    """BaseDataset for using as a template for other Dataset.
    Two ways to
    Arguments:
        loader (Callable): a function to load images.
        target_loader (Callable): a function to load labels and preprocess in
        a preferable format.
    """

    def __init__(
        self,
        loader: Callable,
        target_loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.transform = (lambda x: x) if transform is None else transform
        self.target_transform = (
            (lambda x: x) if target_transform is None else target_transform
        )
        self.loader = (lambda x: x) if loader is None else loader
        self.target_loader = (lambda x: x) if target_loader is None else target_loader
        self.data_dirs, self.label_dirs = None, None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.data_dirs is not None
        ), "Please set via `get_data_label_dirs` or `set_data_label_dirs`."
        assert (
            self.label_dirs is not None
        ), "Please set via `get_data_label_dirs` or `set_data_label_dirs`."

        data, label = self.data_dirs[idx], self.label_dirs[idx]
        data, label = self.loader(data), self.target_loader(label)
        data, label = self.transform(data), self.target_transform(label)
        return data, label

    def __len__(self) -> int:
        return len(self.data_dirs)

    def get_data_label_dirs(self) -> None:
        """Get all data locations and labels. Can inputs more data to further processing."""
        raise NotImplementedError()

    def set_data_label_dirs(self, data_dirs: List[str], label_dirs: List[str]) -> None:
        self.data_dirs, self.label_dirs = data_dirs, label_dirs


class BurstDataset(BaseDataset):
    """Allows for loading all images and labels to RAM for faster access.
    >>> dataset = BurstDataset()
    >>> dataset.load_images()
    >>> dataset.load_labels()
    """

    def __init__(
        self,
        loader: Callable,
        target_loader: Callable = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(loader, target_loader, transform, target_transform)
        self.init_ram = psutil.virtual_memory().percent

    def load_images_fn(self, data_dirs: List[str]) -> List[Any]:
        assert self.loader is not None
        return [self.loader(d) for d in data_dirs]

    def load_images(
        self, load_images_fn: Optional[Callable] = None, num_workers: int = cpu_count()
    ) -> None:
        """Load images and assign identify function to self.loader."""
        if load_images_fn is None:
            # If not define load_images_fn, using for-loop version of loader.
            load_images_fn = self.load_images_fn
        try:
            images = multithread_load_images(
                self.data_dirs, load_images_fn, num_workers
            )
        except AttributeError as e:
            warnings.warn(
                "Cannot use multithread, use only a single thread instead. Error message: {e}.",
                UserWarning,
            )
            images = load_images_fn(self.data_dirs)

        self.data_dirs = images
        self.loader = lambda x: x
        ram_usage = psutil.virtual_memory().percent
        delta = ram_usage - self.init_ram

        WARNING_THRESHOLD = 80.0
        if ram_usage > WARNING_THRESHOLD:
            warnings.warn(
                UserWarning,
                f"RAM usages over {WARNING_THRESHOLD}%. Current around {ram_usage}. "
                f"This BurstDataset consumes around {delta}%.",
            )

    def load_labels(self) -> None:
        """Load labels and preprocess labels, Ex: prepare labels for object detection."""
        # TODO: support multi-thread load and process labels?
        self.label_dirs = [self.target_loader(i) for i in self.label_dirs]
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


def get_mean_std(dataset, burst: bool = True) -> Tuple[float, float]:
    """Get a mean and standard deviation from a dataset.
    Args:
        burst (bool): If True load all data to RAM and calculates a mean and standard deviation.
        Else accumulate all one by one elements to calculate a mean and standard deviation.
    """
    sample, _ = next(iter(dataset))
    assert True if isinstance(sample, torch.Tensor) else False, "Support only PyTorch format."
    assert isinstance(burst, bool)
    if burst:
        dataset = torch.stack([i[0] for i in list(dataset)], dim=0)
        mean = torch.mean(dataset, dim=(0, 2, 3))
        std = torch.std(dataset, dim=(0, 2, 3))
    else:
        # TODO: Fix this.
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


if __name__ == "__main__":
    dataset = BurstImageFolder("~/datasets/CINIC10/train")
    dataset.load_images()
    img, label = next(iter(dataset))
    assert img.size == (32, 32)
