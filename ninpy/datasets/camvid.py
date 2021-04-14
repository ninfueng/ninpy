import os
from typing import List, Tuple

import cv2
import numpy as np
import requests

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


_SEGNET_CAMVID_URL = (
    "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/"
)


def _filter_camvid_path(segnet_path):
    path = segnet_path.split(" ")
    assert len(path) == 2
    basename = os.path.basename(path[0])
    return basename


def camvid_dir(save_txt: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Return list of train (367), val (101), and test (233) labels."""
    assert isinstance(save_txt, bool)
    train_txt = requests.get(os.path.join(_SEGNET_CAMVID_URL, "train.txt"))
    # Skip space on the last line.
    train_txt = train_txt.text.split("\n")[:-1]
    val_txt = requests.get(os.path.join(_SEGNET_CAMVID_URL, "val.txt"))
    val_txt = val_txt.text.split("\n")[:-1]
    test_txt = requests.get(os.path.join(_SEGNET_CAMVID_URL, "test.txt"))
    test_txt = test_txt.text.split("\n")[:-1]

    list_train, list_val, list_test = [], [], []
    for txt in train_txt:
        imgdir = _filter_camvid_path(txt)
        list_train.append(imgdir)

    for txt in val_txt:
        imgdir = _filter_camvid_path(txt)
        list_val.append(imgdir)

    for txt in test_txt:
        imgdir = _filter_camvid_path(txt)
        list_test.append(imgdir)

    if save_txt:
        np.savetxt("train.txt", list_train, fmt="%s")
        np.savetxt("val.txt", list_val, fmt="%s")
        np.savetxt("test.txt", list_test, fmt="%s")
    return list_train, list_val, list_test


# TODO: CamVid loader.
class CamVid(Dataset):
    """CamVid dataset.
    root___images
        |____labels
    """
    def __init__(self, root, loader, mode="train", img_mode="RGB"):
        img_mode = img_mode.upper()
        mode = mode.lower()
        assert mode in ["train", "val", "test"]
        assert img_mode in ["RGB", "BGR"]
        assert isinstance(root, str)

        self.root = root
        self.mode = mode
        self.img_mode = img_mode
        self.loader = loader

        traindir, valdir, testdir = camvid_dir()
        if self.mode == 'train':
            imgdirs = traindir
        if self.mode == 'val':
            imgdirs = valdir
        if self.mode == 'test':
            imgdirs = testdir
        self.img

    def _load_img(self, path, img_mode="RGB"):
        assert isinstance(path, str)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None
        return img

