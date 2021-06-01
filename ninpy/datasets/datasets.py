import glob
import logging
import os
from functools import reduce
from multiprocessing import Pool, cpu_count
from typing import Callable, List

import albumentations as A
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm


def reduce_sum(x, y):
    """Cannot use a lambda function for multi-thread processing.
    This function is required for pickle-able functions."""
    return x + y


def load_images(imgdirs: List[str]) -> List[Callable]:
    """Load images from a list of directories."""
    imgs = []
    for d in imgdirs:
        img = pil_loader(d)
        imgs.append(img)
    return imgs


def multithread_load_images(
    imgdirs: List[str], num_workers=cpu_count()
) -> List[Image.Image]:
    """Multi-processing load images to list."""
    assert isinstance(num_workers, int)
    pool = Pool(num_workers)
    img_per_worker = int(len(imgdirs) / num_workers)
    imgdir_per_workers = [
        imgdirs[i : i + img_per_worker] for i in range(0, len(imgdirs), img_per_worker)
    ]
    imgs = pool.map(load_images, imgdir_per_workers)
    # In case of pickle this function must not be a lambda function.
    imgs = reduce(reduce_sum, imgs)
    return imgs


class BurstImageFolder(ImageFolder):
    """ImageFolder but loading all image into RAM.
    Example:
    >>> traindir = os.path.expanduser("~/datasets/CINIC10/train")
    >>> dataset = BurstImageFolder(traindir)
    >>> dataset.load_imgs()
    """

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

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=pil_loader,
        is_valid_file=None,
        verbose=False,
    ):
        root = os.path.expanduser(root)
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.verbose = verbose

    def load_imgs(self) -> None:
        """TODO: update with multi-processing loading."""
        list_classes = sorted(os.listdir(self.root))
        root = os.path.expanduser(self.root)
        instances = []

        for idx, c in enumerate(tqdm(list_classes)):
            classdir = os.path.join(root, c)
            imgdirs = self.load_img_with_extension(classdir)

            for i in imgdirs:
                img = pil_loader(i)
                item = img, idx
                instances.append(item)

        self.samples = instances
        self.loader = self._identity
        if self.verbose:
            logging.info("Store all image into RAM.")

    def load_img_with_extension(self, path: str) -> List[str]:
        assert isinstance(path, str)
        imgdirs = []
        for e in self.IMG_EXTENSIONS:
            imgdirs += glob.glob(os.path.join(path, "*" + e))
            imgdirs += glob.glob(os.path.join(path, "*" + e.upper()))
        return imgdirs

    @staticmethod
    def _identity(x):
        return x


if __name__ == "__main__":
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
