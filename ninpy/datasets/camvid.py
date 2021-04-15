import os
from typing import List, Tuple

import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# TODO: Camvid loader.
class Camvid(Dataset):
    """CamVid dataset. Support only burst mode. Using `color_map` same as SegNet.
    Labels that are not included in `COLOR_MAP` is assumed to be `11`.
    root___images
        |____labels
    Recommend:
        Using img_mode `BGR` to not waste the converting time.
        Normal shape: (360, 480, 3)
    """

    # https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/camvid.py
    # https://uk.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html
    COLOR_MAP = {
        # Sky
        0: (128, 128, 128),
        # Building
        1: [(0, 128, 64), (128, 0, 0), (64, 192, 0), (64, 0, 64), (192, 0, 128)],
        # Pole
        2: [(192, 192, 128), (0, 0, 64)],
        # Road
        3: [(128, 64, 128), (128, 0, 192), (192, 0, 64)],
        # Pavement
        4: [(0, 0, 192), (64, 192, 128), (128, 128, 192)],
        # Tree
        5: [(128, 128, 0), (192, 192, 0)],
        # SignSymbol
        6: [(192, 128, 128), (128, 128, 64), (0, 64, 64)],
        # Fence
        7: (64, 64, 128),
        # Car
        8: [
            (64, 0, 128),
            (64, 128, 192),
            (192, 128, 192),
            (192, 64, 128),
            # (128, 64, 64), # OtherMoving, check from the SegNet repository does not have.
        ],
        # Pedestrian
        9: [(64, 64, 0), (192, 128, 64), (64, 0, 192), (64, 128, 64)],
        # Bicyclist
        10: [(0, 128, 192), (192, 0, 192)],
        # Void, Other are ignored.
        # 11: (0, 0, 0),
    }

    SEGNET_CAMVID_URL = (
        "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/"
    )

    def __init__(self, root, mode="train", img_mode="RGB"):
        img_mode = img_mode.upper()
        mode = mode.lower()
        assert mode in ["train", "val", "test"]
        assert img_mode in ["RGB", "BGR"]
        assert isinstance(root, str)
        root = os.path.expanduser(root)
        self.mode = mode
        self.img_mode = img_mode
        self.imgs, self.masks = [], []

        traindir, valdir, testdir = self._camvid_dir()
        if self.mode == "train":
            imgdirs = traindir
            assert len(imgdirs) == 367
        if self.mode == "val":
            imgdirs = valdir
            assert len(imgdirs) == 101
        if self.mode == "test":
            imgdirs = testdir
            assert len(imgdirs) == 233

        labelsdirs = [os.path.join(root, "labels", i) for i in imgdirs]
        imgdirs = [os.path.join(root, "images", i) for i in imgdirs]
        for i in imgdirs:
            img = self._load_img(i)
            self.imgs.append(img)

        for i in labelsdirs:
            mask = self._load_mask(i)
            self.masks.append(mask)

    def _load_img(self, path):
        assert isinstance(path, str)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None
        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, path):
        assert isinstance(path, str)
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        x, y = basename.split(".")
        path = os.path.join(dirname, x + "_L" + "." + y)
        mask = cv2.imread(path, cv2.IMREAD_COLOR)
        assert mask is not None
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        cls_masks = np.full((mask.shape[0], mask.shape[1]), 11)
        mask_pth = os.path.expanduser("~/datasets/camvid/labels/0001TP_006840_L.png")
        mask = cv2.imread(mask_pth, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        for i in range(len(self.COLOR_MAP)):
            if i in [1, 2, 3, 4, 5, 6, 8, 9, 10]:
                for j in range(len(self.COLOR_MAP[i])):
                    cls_mask = (
                        cv2.inRange(mask, self.COLOR_MAP[i][j], self.COLOR_MAP[i][j])
                        / 255.0
                    )
                    cls_masks[cls_mask.astype(bool)] = i
            else:
                cls_mask = (
                    cv2.inRange(mask, self.COLOR_MAP[i], self.COLOR_MAP[i]) / 255.0
                )
                cls_masks[cls_mask.astype(bool)] = i

        return cls_masks.astype(np.uint8)

    def _filter_camvid_path(self, segnet_path):
        path = segnet_path.split(" ")
        assert len(path) == 2
        basename = os.path.basename(path[0])
        return basename

    def _camvid_dir(
        self, save_txt: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
        """Loading train, validation, test dataset lists from SegNet repository.
        Return:
            list_train (367):
            list_val (101):
            list_test (233):
        """
        assert isinstance(save_txt, bool)
        train_txt = requests.get(os.path.join(self.SEGNET_CAMVID_URL, "train.txt"))
        # Skip space on the last line.
        train_txt = train_txt.text.split("\n")[:-1]
        val_txt = requests.get(os.path.join(self.SEGNET_CAMVID_URL, "val.txt"))
        val_txt = val_txt.text.split("\n")[:-1]
        test_txt = requests.get(os.path.join(self.SEGNET_CAMVID_URL, "test.txt"))
        test_txt = test_txt.text.split("\n")[:-1]

        list_train, list_val, list_test = [], [], []
        for txt in train_txt:
            imgdir = self._filter_camvid_path(txt)
            list_train.append(imgdir)

        for txt in val_txt:
            imgdir = self._filter_camvid_path(txt)
            list_val.append(imgdir)

        for txt in test_txt:
            imgdir = self._filter_camvid_path(txt)
            list_test.append(imgdir)

        if save_txt:
            np.savetxt("train.txt", list_train, fmt="%s")
            np.savetxt("val.txt", list_val, fmt="%s")
            np.savetxt("test.txt", list_test, fmt="%s")
        return list_train, list_val, list_test


if __name__ == "__main__":

    Camvid("~/datasets/camvid/")
