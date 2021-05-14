"""Basic imagenet functions."""
import os
from multiprocessing import cpu_count
from typing import Callable, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from ninpy.datasets import WrappedCompose

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


from typing import Callable, Tuple, Union


def get_imagenet_albumentations_transforms(
    crop_size: Union[int, Tuple[int, int]], resize_size: Union[int, Tuple[int, int]]
) -> Tuple[Callable, Callable]:

    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    assert len(crop_size) == 2
    assert len(resize_size) == 2

    train_transforms = WrappedCompose(
        [
            A.RandomResizedCrop(crop_size[0], crop_size[1]),
            A.HorizontalFlip(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    val_transforms = WrappedCompose(
        [
            A.Resize(crop_size[0], crop_size[1]),
            A.CenterCrop(resize_size[0], resize_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return train_transforms, val_transforms


def get_imagenet_loaders(
    root: str,
    batch_size: int,
    num_workers: int = cpu_count(),
    crop_size: Union[int, Tuple[int, int]]= 256,
    resize_size: Union[int, Tuple[int, int]] = 224,
    distributed: bool = False,
    train_transforms: Optional[Callable] = None,
    val_transforms: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Get ImageNet loaders by using ImageFolder."""
    assert isinstance(root, str)
    assert isinstance(batch_size, int)
    assert isinstance(num_workers, int)
    assert isinstance(distributed, bool)
    assert isinstance(crop_size, int)
    assert isinstance(resize_size, int)

    root = os.path.expanduser(root)
    traindir = os.path.join(root, "train")
    valdir = os.path.join(root, "val")

    (
        default_train_transforms,
        default_val_transforms,
    ) = get_imagenet_albumentations_transforms(crop_size, resize_size)
    if train_transforms is None:
        train_transforms = default_train_transforms
    if val_transforms is None:
        val_transforms = default_val_transforms

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
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # traindir = os.path.expanduser("~/datasets/CINIC10/train")
    # dataset = ImageFolder(traindir)
    # for x, y in tqdm(dataset):
    #     pass

    # dataset = BurstImageFolder(traindir)
    # dataset.load_imgs()
    # for x, y in tqdm(dataset):
    #     pass
    # print(x, y)
    pass

