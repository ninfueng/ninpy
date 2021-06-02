from functools import reduce
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List

import cv2
import numpy as np
from PIL import Image
from torchvision.datasets.folder import pil_loader

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


def cv2_loader(imgdir: str) -> np.ndarray:
    img = cv2.imread(imgdir, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert img is not None
    return img


def load_images(imgdirs: List[str], load_fn: Callable = pil_loader) -> List[Callable]:
    """Load images from a list of directories.
    This is created as a base function for `multithread_load_images`"""
    imgs = []
    for d in imgdirs:
        img = load_fn(d)
        imgs.append(img)
    return imgs


load_images_cv2 = lambda x: load_images(x, cv2_loader)
load_images_pil = lambda x: load_images(x, pil_loader)


def multithread_load_images(
    imgdirs: List[str],
    load_images_fn: Callable = load_images,
    num_workers: int = cpu_count(),
) -> List[Image.Image]:
    """Multi-processing to load all images to list.
    This created for BurstDataset when loads a large-size dataset."""
    assert isinstance(num_workers, int)
    pool = Pool(num_workers)
    img_per_worker = int(len(imgdirs) / num_workers)
    img_dir_per_workers = [
        imgdirs[i : i + img_per_worker] for i in range(0, len(imgdirs), img_per_worker)
    ]
    imgs = pool.map(load_images_fn, img_dir_per_workers)
    # In case of pickle this function must not be a lambda function.
    imgs = reduce(reduce_sum, imgs)
    return imgs


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

    listimgs = load_images(imgdirs)
    print(listimgs.__len__())
    listimgs = multithread_load_images(imgdirs)
    print(listimgs.__len__())
