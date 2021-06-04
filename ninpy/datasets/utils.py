from functools import reduce
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import pil_loader

__all__ = [
    "IMG_EXTENSIONS",
    "cv2_loader",
    "cv2_load_images",
    "pil_load_images",
    "multithread_load_images",
]

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def reduce_sum(x: Any, y: Any) -> Any:
    """Cannot use a lambda function for multi-thread processing.
    This function is required for pickle-able functions."""
    return x + y


def cv2_loader(img_dir: str, img_mode: str = cv2.COLOR_BGR2RGB) -> np.ndarray:
    assert isinstance(img_dir, str)
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, img_mode)
    assert img is not None
    return img


def pil_load_images(img_dirs: List[str]) -> List[Callable]:
    """Load images from a list of directories.
    This is created as a base function for `multithread_load_images`"""
    return [pil_loader(d) for d in img_dirs]


def cv2_load_images(img_dirs: List[str]) -> List[Callable]:
    """Load images from a list of directories.
    This is created as a base function for `multithread_load_images`
    Need to do this way otherwise cannot multi-processing.
    """
    return [cv2_loader(d) for d in img_dirs]


def multithread_load_images(
    img_dirs: List[str],
    load_images_fn: Callable = pil_load_images,
    num_workers: int = cpu_count(),
) -> List[Image.Image]:
    """Multi-processing to load all images to list.
    This is created for `BurstDataset` especially when loads a large-size dataset."""
    assert isinstance(num_workers, int)
    img_per_worker = int(len(img_dirs) / num_workers)
    if img_per_worker == 0:
        # Number of images less than workers.
        # Use a series loading instead.
        imgs = load_images_fn(img_dirs)
    else:
        pool = Pool(num_workers)
        img_dir_per_workers = [
            img_dirs[i : i + img_per_worker]
            for i in range(0, len(img_dirs), img_per_worker)
        ]
        imgs = pool.map(load_images_fn, img_dir_per_workers)
        # In case of pickle this function must not be a lambda function.
        imgs = reduce(reduce_sum, imgs)
    assert len(imgs) == len(img_dirs)
    return imgs


def get_basic_loader(
    dataset: Dataset, batch_size: int, mode: str, num_workers: int = cpu_count()
) -> DataLoader:
    """Get a loader with two different sets of `kwargs` for train or valid (test)."""
    # TODO: support distributed dataparallel.
    mode = mode.lower()
    assert mode in ["train", "valid", "test"]
    if mode == "train":
        train_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
        loader = DataLoader(dataset, **train_kwargs)
    else:
        valid_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": True,
            "drop_last": False,
        }
        loader = DataLoader(dataset, **valid_kwargs)
    return loader


if __name__ == "__main__":
    import glob
    import os

    root = os.path.expanduser("~/datasets/CINIC10/train")
    list_classes = sorted(os.listdir(root))
    root = os.path.expanduser(root)

    imgdirs = []
    for idx, c in enumerate(list_classes):
        classdir = os.path.join(root, c)
        imgdirs += glob.glob(os.path.join(classdir, "*.png"))

    listimgs = pil_load_images(imgdirs)
    print(listimgs.__len__())
    listimgs = multithread_load_images(imgdirs)
    print(listimgs.__len__())
