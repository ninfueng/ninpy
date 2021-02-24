#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
from multiprocessing import cpu_count
from typing import Callable, Optional, Tuple, Any

import cv2
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import VOCSegmentation, ImageFolder
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .common import show_img_torch


def get_cifar10_transforms():
    r"""Modified from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
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
    r"""Modified from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
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
        num_workers: int = cpu_count(),
        dataset_name: str = 'mnist',
        data_path: str = './dataset',
        train_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None) -> tuple:
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
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
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


def get_imagenet_loaders(
    root: str,
    batch_size: int,
    num_workers: int = cpu_count(),
    distributed: bool = False,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable]  = None):
    r"""Get ImageNet loaders.
    """
    assert isinstance(root, str)
    assert isinstance(batch_size, str)
    assert isinstance(num_workers, str)
    assert isinstance(distributed, bool)

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD)

    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    if val_transforms is None:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = ImageFolder(
        traindir, train_transforms)
    val_dataset = ImageFolder(
        valdir, val_transforms)
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    return train_loader, val_loader


class VOCSegmentationDataset(VOCSegmentation):
    r"""VOCSegmentationDataset.
    Modified from: https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
    """
    def __init__(
        self,
        root: str,
        year: str,
        has_mask: bool,
        transform,
        image_set: str = 'train',
        download: bool = False,
        ):
        super().__init__(
            root=root,
            year=year,
            transform=transform,
            image_set=image_set,
            download=download)
        self.has_mask = has_mask

    @staticmethod
    def _convert_to_segmentation_mask(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        _, colormap = VOCSegmentationDataset.get_classnames_colormap()
        segmentation_mask = np.zeros(
            (height, width, len(colormap)),
            dtype=np.float32)

        for label_index, label in enumerate(colormap):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index: int):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.has_mask:
            mask = cv2.imread(self.masks[index])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            if self.has_mask:
                transformed = self.transform(
                    image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
                return image, mask
            else:
                transformed = self.transform(
                    image=image)
                image = transformed['image']
                return image
        else:
            if self.has_mask:
                return mask
            else:
                return image, mask

    @staticmethod
    def get_classnames_colormap() -> Tuple[Any, Any]:
        r"""
        """
        VOC_CLASSES = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv/monitor",
        ]

        VOC_COLORMAP = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
        return VOC_CLASSES, VOC_COLORMAP


def get_voc2012_loader(
    root: str,
    batch_size: int,
    num_workers: int = cpu_count(),
    distributed: bool = False,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None
    ):
    r"""Modified from: https://pytorch.org/vision/0.8/_modules/torchvision/datasets/voc.html#VOCSegmentation 
    """
    assert isinstance(root, str)
    assert isinstance(batch_size, int)
    assert isinstance(num_workers, int)
    assert isinstance(distributed, bool)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = A.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train_transforms is None:
        train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ])

    if val_transforms is None:
        val_transforms = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            normalize,
            ToTensorV2(),
        ])

    train_dataset = VOCSegmentationDataset(
        has_mask=True,
        root=root,
        year='2012',
        image_set='train',
        download=False,
        transform=train_transforms,
    )
    val_dataset = VOCSegmentationDataset(
        has_mask=False,
        root=root,
        year='2012',
        image_set='val',
        download=False,
        transform=val_transforms,
    )

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    # train_loader, val_loader = get_voc2012_loader(
    #     '/home/ninnart/datasets/VOC',
    #     False, None, 128, 8, False, None, None)

    # test_batch = next(iter(train_loader))
    # print(test_batch)

    root = '/home/ninnart/datasets/VOC'
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = A.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ])
    train_dataset = VOCSegmentationDataset(
        has_mask=True,
        root=root,
        year='2012',
        image_set='train',
        download=False,
        transform=train_transforms,
    )

    img, mask = next(iter(train_dataset))
    show_img_torch(img, True)
    show_img_torch(mask, False)
