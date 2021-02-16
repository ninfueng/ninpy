#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
from multiprocessing import cpu_count

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
NUM_WORKERS = cpu_count()


def get_lenet5_transforms():
    r"""LeNet5 assumes 32x32 size of image,
    therefore need to resize for 28x28 size of image.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


def get_cifar10_transforms():
    r"""Refer: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transform_train, transform_test


def get_cifar100_transforms():
    r"""Refer: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return transform_train, transform_test


def load_toy_dataset(
        num_train_batch: int,
        num_test_batch: int,
        num_workers: int = NUM_WORKERS,
        dataset_name: str = 'mnist',
        data_path: str = './dataset',
        train_transforms: transforms.Compose = None,
        test_transforms: transforms.Compose = None) -> tuple:
    r"""Using torchvision to load the provided dataset online.
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

    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if train_transforms is None:
        train_transforms = transforms.Compose(
            [transforms.ToTensor()])
    if test_transforms is None:
        test_transforms = transforms.Compose(
            [transforms.ToTensor()])

    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.MNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'kmnist':
        train_set = torchvision.datasets.KMNIST(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'emnist':
        train_set = torchvision.datasets.EMNIST(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.CIFAR10(
            root=data_path, train=False,
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.CIFAR100(
            root=data_path, train=False,
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)

    elif dataset_name == 'svhn':
        # The extra-section or extra_set is exist in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=data_path, split='train',
            download=True, transform=train_transforms)
        train_loader = DataLoader(
            train_set, batch_size=num_train_batch,
            shuffle=True, num_workers=num_workers)
        test_set = torchvision.datasets.SVHN(
            root=data_path, split='test',
            download=True, transform=test_transforms)
        test_loader = DataLoader(
            test_set, batch_size=num_test_batch,
            shuffle=False, num_workers=num_workers)
    else:
        raise NotImplementedError(
            'dataset must be in [mnist, fmnist, kmnist, '
            f'emnist, cifar10, cifar100, svhn] only, your input: {dataset}')
    return train_loader, test_loader
