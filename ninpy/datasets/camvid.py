import os
from typing import List, Tuple

import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ninpy.datasets.utils import cv2_loader

# class Camvid(Dataset):
#     """CamVid dataset. Support only burst mode. Using `color_map` same as SegNet.
#     Labels that are not included in `COLOR_MAP` is assumed to be `11`.
#     root___images
#         |____labels
#     Recommend:
#         Using img_mode `BGR` to not waste the converting time.
#         Normal shape for SegNet: (360, 480, 3)
#         Original shape: (720, 960, 3)
#     """

#     # https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/camvid.py
#     # https://uk.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html
#     COLOR_MAP = {
#         # Sky
#         0: (128, 128, 128),
#         # Building
#         1: [(0, 128, 64), (128, 0, 0), (64, 192, 0), (64, 0, 64), (192, 0, 128)],
#         # Pole
#         2: [(192, 192, 128), (0, 0, 64)],
#         # Road
#         3: [(128, 64, 128), (128, 0, 192), (192, 0, 64)],
#         # Pavement
#         4: [(0, 0, 192), (64, 192, 128), (128, 128, 192)],
#         # Tree
#         5: [(128, 128, 0), (192, 192, 0)],
#         # SignSymbol
#         6: [(192, 128, 128), (128, 128, 64), (0, 64, 64)],
#         # Fence
#         7: (64, 64, 128),
#         # Car
#         8: [
#             (64, 0, 128),
#             (64, 128, 192),
#             (192, 128, 192),
#             (192, 64, 128),
#             # (128, 64, 64), # `OtherMoving`, checkout from the SegNet repository.
#         ],
#         # Pedestrian
#         9: [(64, 64, 0), (192, 128, 64), (64, 0, 192), (64, 128, 64)],
#         # Bicyclist
#         10: [(0, 128, 192), (192, 0, 192)],
#         # Void, Other are ignored.
#         # 11: (0, 0, 0),
#     }
#     NUM_CLASSES = 11
#     SEGNET_CAMVID_URL = (
#         "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/"
#     )

#     def __init__(self, root, mode, transforms, img_mode="RGB"):
#         img_mode = img_mode.upper()
#         mode = mode.lower()
#         assert mode in ["train", "val", "test"]
#         assert img_mode in ["RGB", "BGR"]
#         assert isinstance(root, str)
#         root = os.path.expanduser(root)
#         self.mode = mode
#         self.img_mode = img_mode
#         self.imgs, self.masks = [], []
#         self.transforms = transforms

#         traindir, valdir, testdir = self._camvid_dir()
#         if self.mode == "train":
#             imgdirs = traindir
#             assert len(imgdirs) == 367
#         elif self.mode == "val":
#             imgdirs = valdir
#             assert len(imgdirs) == 101
#         elif self.mode == "test":
#             imgdirs = testdir
#             assert len(imgdirs) == 233
#         else:
#             raise ValueError("`mode` can be only `train`, `val`, and `test`.")

#         self.imgdirs = imgdirs
#         labelsdirs = [os.path.join(root, "labels", i) for i in imgdirs]
#         imgdirs = [os.path.join(root, "images", i) for i in imgdirs]

#         pbar = tqdm(imgdirs)
#         pbar.set_description(f"Loading {self.mode} images")
#         for i in pbar:
#             img = self._load_img(i)
#             self.imgs.append(img)

#         pbar = tqdm(labelsdirs)
#         pbar.set_description(f"Loading {self.mode} images")
#         for i in pbar:
#             mask = self._load_mask(i)
#             self.masks.append(mask)

#     def __getitem__(self, idx: int):
#         img, mask = self.imgs[idx], self.masks[idx]

#         if self.transforms is not None:
#             augment = self.transforms(image=img, mask=mask)
#             img, mask = augment["image"], augment["mask"]
#         return img, mask.long()

#     def _load_img(self, path):
#         assert isinstance(path, str)
#         img = cv2.imread(path, cv2.IMREAD_COLOR)
#         assert img is not None
#         if self.img_mode == "RGB":
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img

#     def _load_mask(self, path):
#         assert isinstance(path, str)
#         dirname = os.path.dirname(path)
#         basename = os.path.basename(path)
#         # 0001TP_006840_L.png -> 0001TP_006840_L, .png
#         x, y = basename.split(".")
#         path = os.path.join(dirname, x + "_L" + "." + y)

#         mask = cv2.imread(path, cv2.IMREAD_COLOR)
#         assert mask is not None
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

#         # Default class as Void.
#         cls_masks = np.full((mask.shape[0], mask.shape[1]), 11)
#         for i in range(len(self.COLOR_MAP)):
#             # Class with several sub-classes.
#             if i in [1, 2, 3, 4, 5, 6, 8, 9, 10]:
#                 for j in range(len(self.COLOR_MAP[i])):
#                     cls_mask = (
#                         cv2.inRange(mask, self.COLOR_MAP[i][j], self.COLOR_MAP[i][j])
#                         / 255.0
#                     )
#                     cls_masks[cls_mask.astype(bool)] = i
#             else:
#                 # Class without sub-class.
#                 cls_mask = (
#                     cv2.inRange(mask, self.COLOR_MAP[i], self.COLOR_MAP[i]) / 255.0
#                 )
#                 cls_masks[cls_mask.astype(bool)] = i
#         return cls_masks.astype(np.uint8)

#     def _filter_camvid_path(self, segnet_path):
#         path = segnet_path.split(" ")
#         assert len(path) == 2
#         basename = os.path.basename(path[0])
#         return basename

#     def _camvid_dir(
#         self, save_txt: bool = False
#     ) -> Tuple[List[str], List[str], List[str]]:
#         """Loading train, validation, test dataset lists from SegNet repository.
#         """
#         assert isinstance(save_txt, bool)
#         train_txt = requests.get(os.path.join(self.SEGNET_CAMVID_URL, "train.txt"))
#         # Skip space on the last line.
#         train_txt = train_txt.text.split("\n")[:-1]
#         val_txt = requests.get(os.path.join(self.SEGNET_CAMVID_URL, "val.txt"))
#         val_txt = val_txt.text.split("\n")[:-1]
#         test_txt = requests.get(os.path.join(self.SEGNET_CAMVID_URL, "test.txt"))
#         test_txt = test_txt.text.split("\n")[:-1]

#         list_train, list_val, list_test = [], [], []
#         for txt in train_txt:
#             imgdir = self._filter_camvid_path(txt)
#             list_train.append(imgdir)

#         for txt in val_txt:
#             imgdir = self._filter_camvid_path(txt)
#             list_val.append(imgdir)

#         for txt in test_txt:
#             imgdir = self._filter_camvid_path(txt)
#             list_test.append(imgdir)

#         if save_txt:
#             np.savetxt("train.txt", list_train, fmt="%s")
#             np.savetxt("val.txt", list_val, fmt="%s")
#             np.savetxt("test.txt", list_test, fmt="%s")

#         assert (
#             len(list_train) == 367
#         ), "Maybe SegNet repository was removed. Please check."
#         assert len(list_val) == 101
#         assert len(list_test) == 233
#         return list_train, list_val, list_test

#     def save_imgs_masks(self, save_path: str, resize_size: Tuple[int, int]) -> None:
#         assert isinstance(save_path, str)
#         assert len(resize_size) == 2

#         save_path = os.path.expanduser(save_path)
#         imgdirs = self.imgdirs
#         resized_imgdirs = [os.path.join(save_path, "images", i) for i in imgdirs]
#         os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
#         os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)

#         # For mask
#         x = [i.split(".")[0] for i in imgdirs]
#         y = [i.split(".")[1] for i in imgdirs]
#         resized_maskdirs = [
#             os.path.join(save_path, "labels", i + "_L" + "." + j) for i, j in zip(x, y)
#         ]

#         for p, i in zip(resized_imgdirs, self.imgs):
#             # https://github.com/VITA-Group/FasterSeg/blob/master/tools/utils/img_utils.py#L109
#             resized_img = cv2.resize(i, resize_size, interpolation=cv2.INTER_LINEAR)
#             cv2.imwrite(p, resized_img)

#         for p, i in zip(resized_maskdirs, self.masks):
#             # INTER_NEARST = not change RGB values of images.
#             resized_mask = cv2.resize(i, resize_size, interpolation=cv2.INTER_NEAREST)
#             cv2.imwrite(p, resized_mask)

#     def __len__(self):
#         return len(self.imgs)


# def resize_save_camvid(root: str, save_root: str, resize_size: Tuple[int, int]) -> None:
#     dataset = Camvid(root, "train")
#     dataset.save_imgs_masks(save_root, resize_size)

#     dataset = Camvid(root, "val")
#     dataset.save_imgs_masks(save_root, resize_size)

#     dataset = Camvid(root, "test")
#     dataset.save_imgs_masks(save_root, resize_size)


SEGNET_CAMVID_URL = "https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/"

CAMVID_COLORMAP = {
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
        # (128, 64, 64), # `OtherMoving`, checkout from the SegNet repository.
    ],
    # Pedestrian
    9: [(64, 64, 0), (192, 128, 64), (64, 0, 192), (64, 128, 64)],
    # Bicyclist
    10: [(0, 128, 192), (192, 0, 192)],
    # Void, Other are ignored.
    # 11: (0, 0, 0),
}


def _filter_camvid_path(segnet_path: str) -> str:
    path = segnet_path.split(" ")
    assert len(path) == 2
    basename = os.path.basename(path[0])
    return basename


def camvid_image_names(
    save_txt: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Get train, validation, and test image names from SegNet repository."""
    assert isinstance(save_txt, bool)

    train_txt = requests.get(os.path.join(SEGNET_CAMVID_URL, "train.txt"))
    # Skip space on the last line.
    train_txt = train_txt.text.split("\n")[:-1]
    val_txt = requests.get(os.path.join(SEGNET_CAMVID_URL, "val.txt"))
    val_txt = val_txt.text.split("\n")[:-1]
    test_txt = requests.get(os.path.join(SEGNET_CAMVID_URL, "test.txt"))
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

    assert len(list_train) == 367
    assert len(list_val) == 101
    assert len(list_test) == 233

    if save_txt:
        np.savetxt("train.txt", list_train, fmt="%s")
        np.savetxt("val.txt", list_val, fmt="%s")
        np.savetxt("test.txt", list_test, fmt="%s")
    return list_train, list_val, list_test


def load_img(path: str, img_mode: str = "RGB") -> np.ndarray:
    assert isinstance(path, str)
    img_mode = img_mode.upper()
    assert img_mode in ["RGB", "BGR"]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None
    if img_mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_processed_mask(path: str) -> np.ndarray:
    assert isinstance(path, str)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    # 0001TP_006840_L.png -> 0001TP_006840_L, .png
    x, y = basename.split(".")
    path = os.path.join(dirname, x + "_L" + "." + y)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    return mask


def load_mask(path: str) -> np.ndarray:
    assert isinstance(path, str)
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    # 0001TP_006840_L.png -> 0001TP_006840_L, .png
    x, y = basename.split(".")
    path = os.path.join(dirname, x + "_L" + "." + y)

    mask = cv2.imread(path, cv2.IMREAD_COLOR)
    assert mask is not None
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Default class as Void.
    cls_masks = np.full((mask.shape[0], mask.shape[1]), 11)
    for i in range(len(CAMVID_COLORMAP)):
        # Class with several sub-classes.
        if i in [1, 2, 3, 4, 5, 6, 8, 9, 10]:
            for j in range(len(CAMVID_COLORMAP[i])):
                cls_mask = (
                    cv2.inRange(
                        mask, CAMVID_COLORMAP[i][j], CAMVID_COLORMAP[i][j]
                    )
                    / 255.0
                )
                cls_masks[cls_mask.astype(bool)] = i
        else:
            # Class without sub-class.
            cls_mask = (
                cv2.inRange(mask, CAMVID_COLORMAP[i], CAMVID_COLORMAP[i])
                / 255.0
            )
            cls_masks[cls_mask.astype(bool)] = i
    return cls_masks.astype(np.uint8)


def save_imgs_masks(
    load_path: str, save_path: str, resize_size: Tuple[int, int]
) -> None:
    assert isinstance(load_path, str)
    assert isinstance(save_path, str)
    assert len(resize_size) == 2

    load_path = os.path.expanduser(load_path)
    save_path = os.path.expanduser(save_path)
    train_names, val_names, test_names = camvid_image_names()
    names = train_names + val_names + test_names
    x, y = [i.split(".")[0] for i in names], [i.split(".")[1] for i in names]

    load_imgdirs = [os.path.join(load_path, "images", n) for n in names]
    load_maskdirs = [os.path.join(load_path, "labels", n) for n in names]
    imgs = [cv2_loader(i) for i in load_imgdirs]
    masks = [load_mask(i) for i in load_maskdirs]

    resized_imgdirs = [os.path.join(save_path, "images", i) for i in names]
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)

    # For mask
    resized_maskdirs = [
        os.path.join(save_path, "labels", i + "_L" + "." + j)
        for i, j in zip(x, y)
    ]

    pbar = tqdm(zip(resized_imgdirs, imgs), total=len(imgs))
    pbar.set_description("Save images")
    for p, i in pbar:
        # https://github.com/VITA-Group/FasterSeg/blob/master/tools/utils/img_utils.py#L109
        resized_img = cv2.resize(
            i, resize_size, interpolation=cv2.INTER_LINEAR
        )
        cv2.imwrite(p, resized_img)

    pbar = tqdm(zip(resized_maskdirs, masks), total=len(masks))
    pbar.set_description("Save masks")
    for p, i in pbar:
        # INTER_NEARST = not change RGB values of images.
        resized_mask = cv2.resize(
            i, resize_size, interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(p, resized_mask)


class Camvid(Dataset):
    """CamVid dataset. Support only burst mode. Using `color_map` same as SegNet.
    Labels that are not included in `COLOR_MAP` is assumed to be `11`.
    root___images
        |____labels
    Recommend:
        Using img_mode `BGR` to not waste the converting time.
        Normal shape for SegNet: (360, 480, 3)
        Original shape: (720, 960, 3)
    """

    # https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/camvid.py
    # https://uk.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html

    NUM_CLASSES = 11

    def __init__(
        self, root, mode, process_mask, transforms=None, img_mode="RGB"
    ):
        img_mode = img_mode.upper()
        mode = mode.lower()
        assert mode in ["train", "val", "test"]
        assert isinstance(root, str)
        root = os.path.expanduser(root)
        self.mode = mode
        self.img_mode = img_mode
        self.imgs, self.masks = [], []
        self.transforms = transforms

        traindir, valdir, testdir = camvid_image_names()
        if self.mode == "train":
            imgdirs = traindir
            assert len(imgdirs) == 367
        elif self.mode == "val":
            imgdirs = valdir
            assert len(imgdirs) == 101
        elif self.mode == "test":
            imgdirs = testdir
            assert len(imgdirs) == 233
        else:
            raise ValueError("`mode` can be only `train`, `val`, and `test`.")

        self.imgdirs = imgdirs
        labelsdirs = [os.path.join(root, "labels", i) for i in imgdirs]
        imgdirs = [os.path.join(root, "images", i) for i in imgdirs]

        pbar = tqdm(imgdirs)
        pbar.set_description(f"Loading {self.mode} images")
        for i in pbar:
            img = load_img(i)
            self.imgs.append(img)

        pbar = tqdm(labelsdirs)
        pbar.set_description(f"Loading {self.mode} masks")
        for i in pbar:
            if process_mask:
                mask = load_mask(i)
            else:
                mask = load_processed_mask(i)
            self.masks.append(mask)

    def __getitem__(self, idx: int):
        img, mask = self.imgs[idx], self.masks[idx]

        if self.transforms is not None:
            augment = self.transforms(image=img, mask=mask)
            img, mask = augment["image"], augment["mask"]
        return img, mask.long()

    def __len__(self):
        return len(self.imgs)


# def resize_save_camvid(root: str, save_root: str, resize_size: Tuple[int, int]) -> None:
#     dataset = Camvid(root, "train")
#     dataset.save_imgs_masks(save_root, resize_size)

#     dataset = Camvid(root, "val")
#     dataset.save_imgs_masks(save_root, resize_size)

#     dataset = Camvid(root, "test")
#     dataset.save_imgs_masks(save_root, resize_size)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # dataset = Camvid("~/datasets/camvid", "train")
    # plt.imshow(dataset.imgs[0])
    # plt.show()
    # plt.imshow(dataset.masks[0])
    # plt.show()
    # resize_save_camvid("~/datasets/camvid", "~/datasets/camvid256", (256, 256))
    # resize_save_camvid("~/datasets/camvid", "~/datasets/camvid480_360", (480, 360))

    dataset = Camvid("~/datasets/camvid480_360", "train", False)
    plt.imshow(dataset.imgs[0])
    plt.show()

    plt.imshow(dataset.masks[0])
    plt.show()
