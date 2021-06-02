import glob
import os
from multiprocessing import cpu_count
from typing import Tuple

import albumentations as A
import cv2
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder

from ninpy.datasets.augment import CINIC10_MEAN, CINIC10_STD, get_cinic10_transforms

__all__ = ["get_cinic10_basic", "Cinic10", "get_cinic10_loaders"]


def get_cinic10_basic(
    root: str = "~/datasets/CINIC10",
) -> Tuple[ImageFolder, ImageFolder, ImageFolder]:
    """Get CINIC10 dataset with official settings from CINIC github repository.
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


class Cinic10(Dataset):
    """Load CINIC10 all images and stores into RAM for fast processing.
    Note that supports only transforms from torchvision only.
    >>> Cinic10('~/datasets/CINIC10', mode='train', transforms=basic_transform)
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

    def __init__(self, root: str, mode: str, transforms=None) -> None:
        super().__init__()
        self.transforms = transforms
        root = os.path.expanduser(root)
        self.images, self.labels = [], []
        self.mode = mode.lower()

        if self.mode == "train":
            img_dir = os.path.join(root, "train")
        elif self.mode == "valid":
            img_dir = os.path.join(root, "valid")
        elif self.mode == "test":
            img_dir = os.path.join(root, "test")
        else:
            raise ValueError(
                f"mode should be in `train`, `valid`, or `test`, your mode: {self.mode}"
            )

        for k in self.CLASSES.keys():
            search_glob = os.path.join(img_dir, k, "*.png")
            img_dirs = glob.glob(search_glob)

            for i in img_dirs:
                img = cv2.imread(i, cv2.IMREAD_COLOR)
                assert img is not None, f"Cannot find image from {i}."
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Keep every image to RAM for faster access.
                self.images.append(img)
            self.labels += [self.CLASSES[k] for _ in img_dirs]

        assert (
            len(self.images) == 90_000
        ), f"Some images are missing. Found {len(self.image)} images."
        assert (
            len(self.labels) == 90_000
        ), f"Some labels are missing. Found {len(self.labels)} images."

    def __getitem__(self, idx: int):
        img, label = self.images[idx], self.labels[idx]

        if self.transforms is not None:
            img = transforms(image=img)["image"]
        else:
            if self.mode == "train":
                transform = A.Compose(
                    [
                        A.RandomCrop(32, padding=4),
                        A.RandomHorizontalFlip(),
                        A.Normalize(CINIC10_MEAN, CINIC10_STD),
                        ToTensorV2(),
                    ]
                )

            else:
                transform = A.Compose(
                    [A.Normalize(CINIC10_MEAN, CINIC10_STD), ToTensorV2()]
                )
            img = transform(image=img)["image"]
        return img, label

    def __len__(self) -> int:
        return len(self.images)


def get_cinic10_loaders(root: str, batch_size: int, num_workers: int = cpu_count()):
    assert isinstance(root, str)
    assert isinstance(num_workers, int)
    assert isinstance(batch_size, int)

    root = os.path.expanduser(root)
    train_set = Cinic10(root, mode="train")
    val_set = Cinic10(root, mode="valid")
    test_set = Cinic10(root, mode="test")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from tqdm import tqdm

    train_dataset, val_dataset, test_dataset = get_cinic10_basic()
    print("Testing load using ImageFolder.")
    for i, j in tqdm(val_dataset):
        pass

    print("Testing load from RAM.")
    train_dataset = Cinic10("~/datasets/CINIC10", mode="test")
    for a, b in tqdm(train_dataset):
        pass
