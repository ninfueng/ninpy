#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: changing all loading from PIL and transform to cv2 and albumentation.
TODO: supporting the parallel computing.
@author: Ninnart Fuengfusin
"""
import os
from multiprocessing import cpu_count
from typing import Callable, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation, ImageFolder
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

STD_MNIST_MEAN = (0.1307)
STD_MNIST_STD = (0.3081)
# From: https://github.com/quark0/darts/blob/master/cnn/utils.py
STD_CIFAR10_MEAN = (0.49139968, 0.48215827, 0.44653124) #(0.4914, 0.4822, 0.4465)
STD_CIFAR10_STD = (0.24703233, 0.24348505, 0.26158768) #(0.2023, 0.1994, 0.2010)
STD_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
STD_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
STD_IMAGENET_MEAN = (0.485, 0.456, 0.406)
STD_IMAGENET_STD = (0.229, 0.224, 0.225)

def show_img_torch(x: torch.Tensor, denormalize: bool = False) -> None:
    r"""Show an image from torch format with an option to denormalize imagenet normalized image.
    From: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/4
    >>> show_img_torch(torch.zeros(3, 224, 224), False)
    """
    assert isinstance(denormalize, bool)
    assert len(x.shape) == 3
    if denormalize:
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        x = inv_normalize(x)
    x = x.transpose(0, 2).detach().cpu().numpy()
    plt.imshow(x)
    plt.show()


def get_mnist_transforms() -> Tuple[Callable]:
    r"""Modified: https://github.com/itayhubara/BinaryNet.pytorch/blob/master/main_mnist.py
    """
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                STD_MNIST_MEAN, STD_MNIST_STD)])
    return transform


def get_cifar10_transforms() -> Tuple[Callable, Callable]:
    r"""
    Modified:
        https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(STD_CIFAR10_MEAN, STD_CIFAR10_STD)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(STD_CIFAR10_MEAN, STD_CIFAR10_STD)])
    return transform_train, transform_test


def get_cifar100_transforms() -> Tuple[Callable, Callable]:
    r"""
    Modified:
        https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(STD_CIFAR100_MEAN, STD_CIFAR100_STD)])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(STD_CIFAR100_MEAN, STD_CIFAR100_STD)])
    return transform_train, transform_test


def load_toy_dataset(
        num_train_batch: int,
        num_test_batch: int,
        num_workers: int = cpu_count(),
        dataset_name: str = 'mnist',
        data_path: str = './dataset',
        drop_last: bool = True,
        train_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None
        ) -> Tuple[Callable, Callable]:

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
    assert isinstance(drop_last, bool)

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
        test_set = torchvision.datasets.MNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)

    elif dataset_name == 'fmnist':
        train_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)

    elif dataset_name == 'emnist':
        train_set = torchvision.datasets.EMNIST(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False,
            download=True, transform=test_transforms)

    elif dataset_name == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        test_set = torchvision.datasets.CIFAR10(
            root=data_path, train=False,
            download=True, transform=test_transforms)

    elif dataset_name == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True,
            download=True, transform=train_transforms)
        test_set = torchvision.datasets.CIFAR100(
            root=data_path, train=False,
            download=True, transform=test_transforms)

    elif dataset_name == 'svhn':
        # The extra-section or extra_set is exist in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=data_path, split='train',
            download=True, transform=train_transforms)
        test_set = torchvision.datasets.SVHN(
            root=data_path, split='test',
            download=True, transform=test_transforms)
    else:
        raise NotImplementedError(
            'dataset must be in [mnist, fmnist, kmnist, '
            f'emnist, cifar10, cifar100, svhn] only, your input: {dataset_name}')

    train_loader = DataLoader(
        train_set, batch_size=num_train_batch, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    test_loader = DataLoader(
        test_set, batch_size=num_test_batch, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return train_loader, test_loader


def get_imagenet_loaders(
    root: str,
    batch_size: int,
    num_workers: int = cpu_count(),
    crop_size: int = 256,
    resize_size: int = 224,
    distributed: bool = False,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable]  = None):
    r"""Get ImageNet loaders by using ImageFolder.
    """
    assert isinstance(root, str)
    assert isinstance(batch_size, str)
    assert isinstance(num_workers, str)
    assert isinstance(distributed, bool)
    assert isinstance(crop_size, int)
    assert isinstance(resize_size, int)

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(
        mean=STD_IMAGENET_MEAN, std=STD_IMAGENET_STD)

    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    if val_transforms is None:
        val_transforms = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(resize_size),
            transforms.ToTensor(),
            normalize])

    train_dataset = ImageFolder(traindir, train_transforms)
    val_dataset = ImageFolder(valdir, val_transforms)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
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


# TODO: VOCAugSegmentationDataset BASE_AUG = 'benchmark_RELEASE/dataset'
class VOCSegmentationDataset(Dataset):
    r"""TODO: downloading dataset and aug dataset.
    Modified:
        https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
        https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/scripts/prepare_pascal.py
    """
    VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
    'tv/monitor', 'ambigious']

    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
        [128, 192, 0], [0, 64, 128],
    ]
    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable]) -> None:
        assert isinstance(root, str)
        assert isinstance(train, bool)

        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform

        BASE_VOC = 'VOCdevkit/VOC2012/'
        base_dir = os.path.join(self.root, BASE_VOC)
        mask_dir = os.path.join(base_dir, 'SegmentationClass')
        image_dir = os.path.join(base_dir, 'JPEGImages')
        label_dir = os.path.join(base_dir, 'ImageSets/Segmentation')

        if train:
            label_txt = os.path.join(label_dir, 'train.txt')
        else:
            label_txt = os.path.join(label_dir, 'val.txt')

        labels = np.loadtxt(label_txt, dtype=str).tolist()
        self.images = [os.path.join(image_dir, l + '.jpg') for l in labels]
        if train:
            self.masks = [os.path.join(mask_dir, l + '.png') for l in labels]
            assert len(self.images) == len(self.masks)
        else:
            self.mask = None

    def __getitem__(self, index: int):
        print(self.images[index])
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = self.cvt2mask(mask)

        if self.transform is not None:
            if self.train:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
                return image, mask
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
                return image
        else:
            if self.train:
                transform = A.Compose([
                    A.Resize(520, 520),
                    A.RandomCrop(480, 480),
                    A.HorizontalFlip(),
                    A.Normalize(mean=STD_IMAGENET_MEAN, std=STD_IMAGENET_STD),
                    ToTensorV2()])
                transformed = transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
                return image, mask
            else:
                transform = A.Compose([
                    A.Resize(520, 520),
                    A.Crop(480, 480),
                    A.HorizontalFlip(),
                    A.Normalize(mean=STD_IMAGENET_MEAN, std=STD_IMAGENET_STD),
                    ToTensorV2()])
                transformed = transform(image=image, mask=mask)
                image = transformed['image']
                return image

    def cvt2mask(self, mask: np.ndarray) -> np.ndarray:
        r"""Convert a mask to class array.
        """
        height, width = mask.shape[:2]
        colormap = self.VOC_COLORMAP
        segmentation_mask = np.zeros((height, width, len(colormap)), dtype=np.float32)

        for label_index, label in enumerate(colormap):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask


if __name__ == '__main__':
    # train_loader, val_loader = get_voc2012_loader(
    #     '/home/ninnart/datasets/VOC',
    #     False, None, 128, 8, False, None, None)

    # test_batch = next(iter(train_loader))
    # print(test_batch)

    train_transforms = A.Compose([
            A.Resize(520, 520),
            A.RandomCrop(480, 480),
            A.HorizontalFlip(),
            A.Normalize(mean=STD_IMAGENET_MEAN, std=STD_IMAGENET_STD),
            ToTensorV2()])

    root = '~/datasets/VOC'
    root = os.path.expanduser(root)
    train_dataset = VOCSegmentationDataset(root=root, train=True, transform=train_transforms)

    img, mask = next(iter(train_dataset))
    show_img_torch(img, True)