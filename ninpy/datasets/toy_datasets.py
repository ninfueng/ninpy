#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: changing all loading from PIL and transform to cv2 and albumentation.
TODO: supporting the parallel computing.
@author: Ninnart Fuengfusin
"""
import os
from multiprocessing import cpu_count
from typing import Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from ninpy.datasets.augment import IMAGENET_MEAN, IMAGENET_STD


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
            root=data_path, train=True, download=True, transform=train_transforms
        )
        test_set = torchvision.datasets.MNIST(
            root=data_path, train=False, download=True, transform=test_transforms
        )

    elif dataset_name == "fmnist":
        train_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=True, download=True, transform=train_transforms
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False, download=True, transform=test_transforms
        )

    elif dataset_name == "emnist":
        train_set = torchvision.datasets.EMNIST(
            root=data_path, train=True, download=True, transform=train_transforms
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=data_path, train=False, download=True, transform=test_transforms
        )

    elif dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transforms
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=test_transforms
        )

    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transforms
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=test_transforms
        )

    elif dataset_name == "svhn":
        # The extra dataset does not include in this dataset.
        train_set = torchvision.datasets.SVHN(
            root=data_path, split="train", download=True, transform=train_transforms
        )
        test_set = torchvision.datasets.SVHN(
            root=data_path, split="test", download=True, transform=test_transforms
        )
    else:
        raise NotImplementedError(
            "dataset must be in [mnist, fmnist, kmnist, "
            f"emnist, cifar10, cifar100, svhn] only, your input: {dataset_name}"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=num_train_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    # Not drop last for the evaluation.
    test_loader = DataLoader(
        test_set,
        batch_size=num_test_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


# TODO: VOCAugSegmentationDataset BASE_AUG = 'benchmark_RELEASE/dataset'
class VOCSegmentationDataset(Dataset):
    """TODO: downloading dataset and aug dataset.
    Modified:
        https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
        https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/scripts/prepare_pascal.py
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
        "potted-plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "ambigious",
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

    def __init__(self, root: str, train: bool, transform: Optional[Callable]) -> None:
        assert isinstance(root, str)
        assert isinstance(train, bool)

        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        BASE_VOC = "VOCdevkit/VOC2012/"
        base_dir = os.path.join(self.root, BASE_VOC)
        mask_dir = os.path.join(base_dir, "SegmentationClass")
        image_dir = os.path.join(base_dir, "JPEGImages")
        label_dir = os.path.join(base_dir, "ImageSets/Segmentation")

        if train:
            label_txt = os.path.join(label_dir, "train.txt")
        else:
            label_txt = os.path.join(label_dir, "val.txt")

        labels = np.loadtxt(label_txt, dtype=str).tolist()
        self.images = [os.path.join(image_dir, l + ".jpg") for l in labels]
        if train:
            self.masks = [os.path.join(mask_dir, l + ".png") for l in labels]
            assert len(self.images) == len(self.masks)
        else:
            self.mask = None

    def __getitem__(self, index: int):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = self.cvt2mask(mask)

        if self.transform is not None:
            if self.train:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                return image, mask
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]
                return image
        else:
            if self.train:
                transform = A.Compose(
                    [
                        A.Resize(520, 520),
                        A.RandomCrop(480, 480),
                        A.HorizontalFlip(),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
                transformed = transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                return image, mask
            else:
                transform = A.Compose(
                    [
                        A.Resize(520, 520),
                        A.Crop(480, 480),
                        A.HorizontalFlip(),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
                transformed = transform(image=image, mask=mask)
                image = transformed["image"]
                return image

    def cvt2mask(self, mask: np.ndarray) -> np.ndarray:
        r"""Convert a mask to class array."""
        height, width = mask.shape[:2]
        colormap = self.VOC_COLORMAP
        segmentation_mask = np.zeros((height, width, len(colormap)), dtype=np.float32)

        for label_index, label in enumerate(colormap):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)
        return segmentation_mask


if __name__ == "__main__":
    from ninpy.debug import show_torch_image

    # train_loader, val_loader = get_voc2012_loader(
    #     '/home/ninnart/datasets/VOC',
    #     False, None, 128, 8, False, None, None)

    # test_batch = next(iter(train_loader))
    # print(test_batch)

    train_transforms = A.Compose(
        [
            A.Resize(520, 520),
            A.RandomCrop(480, 480),
            A.HorizontalFlip(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    root = "~/datasets/VOC"
    root = os.path.expanduser(root)
    train_dataset = VOCSegmentationDataset(
        root=root, train=True, transform=train_transforms
    )

    img, mask = next(iter(train_dataset))
    show_torch_image(img, True)
