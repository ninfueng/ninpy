import glob
import os
from typing import Callable, List, Optional, Tuple

from torchvision.datasets import ImageFolder

from ninpy.datasets import BurstDataset
from ninpy.datasets.augment import get_cinic10_transforms

__all__ = ["get_cinic10_basic", "get_cinic10_loaders", "CINIC10", "BurstCINIC10"]


def get_cinic10_basic(
    root: str = "~/datasets/CINIC10",
) -> Tuple[ImageFolder, ImageFolder, ImageFolder]:
    """Get CINIC10 dataset with official settings from CINIC10 github repository.
    Using ImageFolder and unit variance and zero mean normalization only.

    Expect folders in this format:
        root
        |__train
            |__airplane
            ...
        |__test
        |__valid
    """
    root = os.path.expanduser(root)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "valid")
    test_dir = os.path.join(root, "test")
    transforms = get_cinic10_transforms()

    train_dataset = ImageFolder(train_dir, transform=transforms)
    val_dataset = ImageFolder(val_dir, transform=transforms)
    test_dataset = ImageFolder(test_dir, transform=transforms)
    return train_dataset, val_dataset, test_dataset


class CINIC10(BurstDataset):
    """CINIC10 dataset with a burst mode support.
    ImageFolder 29 seconds, CINIC10 20 seconds, and Burst mode 8 seconds.
    >>> train_transform, valid_transform = get_cinic10_albumentations_transforms()
    >>> dataset = BurstCINIC10("~/datasets/CINIC10", cv2_loader, transform=train_transform)
    >>> dataset.load_images(cv2_load_images)
    >>> dataset.load_labels()
    """

    CLASSES = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    NUM_CLASSES = len(CLASSES)

    def __init__(
        self,
        root: str,
        loader: Callable,
        target_loader: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mode: str = "train",
    ) -> None:
        mode = mode.lower()
        assert mode in ["train", "test", "valid"]
        self.mode = mode
        # Need to add all child-attributes before initializes super-class.
        super().__init__(root, loader, target_loader, transform, target_transform)

    def get_data_dirs_labels(self) -> Tuple[List[str], List[int]]:
        data_dir = os.path.join(self.root, self.mode)
        data_dirs, labels = [], []

        for k, v in self.CLASSES.items():
            tmp_dir = glob.glob(os.path.join(data_dir, k, "*.png"))
            data_dirs += tmp_dir
            tmp_label = [v for _ in tmp_dir]
            labels += tmp_label
        assert len(data_dirs) == len(labels) == 90_000
        return data_dirs, labels


if __name__ == "__main__":
    from ninpy.datasets.augment import get_cinic10_albumentations_transforms
    from ninpy.datasets.utils import cv2_load_images, cv2_loader
    from ninpy.torch2 import torch2np

    train_dataset, val_dataset, test_dataset = get_cinic10_basic()
    print("Load using a default ImageFolder.")
    for img, label in train_dataset:
        pass

    train_transform, valid_transform = get_cinic10_albumentations_transforms()
    dataset = CINIC10(
        "~/datasets/CINIC10", cv2_loader, transform=train_transform, mode="train"
    )
    print("Load using a CINIC10.")
    for img, label in dataset:
        pass
    img = torch2np(img)

    dataset.load_images(cv2_load_images)
    dataset.load_labels()
    print("Load using a burst CINIC10.")
    for img, label in dataset:
        pass
    img = torch2np(img)
