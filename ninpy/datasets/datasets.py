import glob
import logging
import os
from functools import reduce
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Optional

import albumentations as A
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

from ninpy.datasets.utils import IMG_EXTENSIONS, multithread_load_images


class BurstImageFolder(ImageFolder):
    """ImageFolder instead of load images at a runtime.
    Load all images to a RAM to faster access.
    Example:
    >>> traindir = os.path.expanduser("~/datasets/CINIC10/train")
    >>> dataset = BurstImageFolder(traindir)
    >>> dataset.load_imgs()
    """

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=pil_loader,
        is_valid_file: Optional[bool] = None,
        verbose: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.classes = sorted(os.listdir(self.root))
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.verbose = verbose
        self.samples = self.load_imgs()
        self.loader = lambda x: x

    def load_imgs(self) -> None:
        imgdirs = [self.get_imgdirs(os.path.join(self.root, c)) for c in self.classes]
        numlabels = [len(i) for i in imgdirs]
        labels = []
        for idx, n in enumerate(numlabels):
            for _ in range(n):
                labels.append(idx)
        imgdirs = reduce(lambda x, y: x + y, imgdirs)
        imgs = multithread_load_images(imgdirs)
        if self.verbose:
            logging.info("Store all image into RAM.")
        return [(i, d) for i, d in zip(imgs, labels)]

    def get_imgdirs(self, path: str) -> List[str]:
        assert isinstance(path, str)
        imgdirs = []
        for e in IMG_EXTENSIONS:
            imgdirs += glob.glob(os.path.join(path, "*" + e))
            imgdirs += glob.glob(os.path.join(path, "*" + e.upper()))
        return imgdirs


if __name__ == "__main__":
    traindir = os.path.expanduser("~/datasets/CINIC10/train")
    dataset = BurstImageFolder(traindir)
    dataset.load_imgs()
    img, label = next(iter(dataset))
    print(label)
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()
