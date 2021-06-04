"""Collection of online augmentation functions and objects."""
from typing import Callable, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR10_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR10_STD = (0.24703233, 0.24348505, 0.26158768)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ClassifyCompose(A.Compose):
    """Designed to make albumentations operate with template Dataset provided by PyTorch
    >>> compose = ClassifyCompose([ToTensorV2()])
    >>> output = compose(np.zeros((32, 32, 3)))
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, image: np.ndarray, **kwargs) -> torch.Tensor:
        aug = super().__call__(image=image, **kwargs)
        return aug["image"]


class SegmentCompose(A.Compose):
    """Designed to make albumentations operate with template Dataset provided by PyTorch
    >>> compose = SegmentCompose([ToTensorV2()])
    >>> output, mask = compose(np.zeros((32, 32, 3)), np.zeros((32, 32, 20)))
    TODO: Test case and check that can be used.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aug = super().__call__(image=image, mask=mask, **kwargs)
        return aug["image"], aug["mask"]


def get_mnist_transforms() -> Tuple[Callable]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    return transform


def get_cifar10_transforms() -> Tuple[Callable, Callable]:
    """Random pad four pixels around an image."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    return train_transform, test_transform


def get_cifar100_transforms() -> Tuple[Callable, Callable]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    return train_transform, test_transform


def get_imagenet_transforms(
    crop_size: Union[int, Tuple[int, int]], resize_size: Union[int, Tuple[int, int]]
) -> Tuple[Callable, Callable]:
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    assert len(crop_size) == 2 and len(resize_size) == 2

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(resize_size[0]),
            transforms.RandomHorizontalFlip(crop_size[0]),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(resize_size[0]),
            transforms.CenterCrop(crop_size[0]),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transforms, val_transforms


def get_imagenet_albumentations_transforms(
    crop_size: Union[int, Tuple[int, int]], resize_size: Union[int, Tuple[int, int]]
) -> Tuple[Callable, Callable]:

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    assert len(crop_size) == 2 and len(resize_size) == 2
    assert crop_size[0] > resize_size[0]

    train_transforms = ClassifyCompose(
        [
            A.RandomResizedCrop(resize_size[0], resize_size[1]),
            A.HorizontalFlip(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    val_transforms = ClassifyCompose(
        [
            A.Resize(crop_size[0], crop_size[1]),
            A.CenterCrop(resize_size[0], resize_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return train_transforms, val_transforms


def get_cinic10_transforms() -> Callable:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CINIC10_MEAN, std=CINIC10_STD),
        ]
    )
    return transform


def get_cinic10_albumentations_transforms() -> Tuple[Callable, Callable]:
    """With `ClassifyCompose`, not need for aug['mask']."""
    train_transforms = ClassifyCompose(
        [
            A.PadIfNeeded(34, 34),
            A.RandomCrop(32, 32),
            A.HorizontalFlip(),
            A.Normalize(CINIC10_MEAN, CINIC10_STD),
            ToTensorV2(),
        ]
    )
    val_transforms = transforms.Compose(
        [A.Normalize(CIFAR100_MEAN, CINIC10_STD), ToTensorV2()]
    )
    return train_transforms, val_transforms
