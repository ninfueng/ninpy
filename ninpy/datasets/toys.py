#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
from multiprocessing import cpu_count
from typing import Callable, Optional, Tuple

import torchvision
import torchvision.transforms as transforms

from ninpy.datasets.utils import get_basic_loader

__all__ = ["load_toy_datasets"]


def load_toy_dataset(
    num_train_batch: int,
    num_test_batch: int,
    num_workers: int = cpu_count(),
    dataset_name: str = "mnist",
    data_path: str = "./dataset",
    drop_last: bool = True,
    train_transforms: Optional[Callable] = None,
    test_transforms: Optional[Callable] = None,
) -> Tuple[Callable, Callable]:

    """Using torchvision to load the provided dataset online.
    Can using with pre-defined transform function with the predefind mean and std.
    Using transform_list=normalize_transforms(CIFAR10_MEAN, CIFAR10_STD)
    Args:
        num_train_batch (int): number of training batch.
        num_test_batch (int): number of test batch.
    """
    assert isinstance(num_train_batch, int)
    assert isinstance(num_test_batch, int)
    assert isinstance(num_workers, int)
    assert isinstance(dataset_name, str)
    assert isinstance(data_path, str)
    assert isinstance(drop_last, bool)

    data_path = os.path.expanduser(data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if train_transforms is None:
        train_transforms = transforms.Compose([transforms.ToTensor()])
    if test_transforms is None:
        test_transforms = transforms.Compose([transforms.ToTensor()])

    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        train_set = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=test_transforms,
        )

    elif dataset_name == "fmnist":
        train_set = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=False,
            download=True,
            transform=test_transforms,
        )

    elif dataset_name == "emnist":
        train_set = torchvision.datasets.EMNIST(
            root=data_path,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path,
            train=False,
            download=True,
            transform=test_transforms,
        )

    elif dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=test_transforms,
        )

    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=data_path,
            train=True,
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_path,
            train=False,
            download=True,
            transform=test_transforms,
        )

    elif dataset_name == "svhn":
        # The extra dataset does not include in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=data_path,
            split="train",
            download=True,
            transform=train_transforms,
        )
        test_set = torchvision.datasets.SVHN(
            root=data_path,
            split="test",
            download=True,
            transform=test_transforms,
        )
    else:
        raise NotImplementedError(
            "dataset must be in [mnist, fmnist, kmnist, "
            f"emnist, cifar10, cifar100, svhn] only, your input: {dataset_name}"
        )
    train_loader = get_basic_loader(
        train_set, num_train_batch, "train", num_workers
    )
    test_loader = get_basic_loader(
        test_set, num_test_batch, "test", num_workers
    )
    return train_loader, test_loader
