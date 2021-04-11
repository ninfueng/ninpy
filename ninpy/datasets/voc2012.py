"""Modified from: https://github.com/zhanghang1989/PyTorch-Encoding
"""
import os
import random

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

__all__ = ["VOCSegmentation", "VOCAugSegmentation"]


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        mode=None,
        transform=None,
        target_transform=None,
        base_size=520,
        crop_size=480,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        if self.mode == "train":
            print(
                "BaseDataset: base_size {}, crop_size {}".format(base_size, crop_size)
            )

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.0))
        y1 = int(round((h - outsize) / 2.0))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        w, h = img.size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


class VOCSegmentation(BaseDataset):
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted-plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "ambigious",
    ]
    NUM_CLASS = 21
    BASE_DIR = "VOCdevkit/VOC2012"

    def __init__(
        self,
        root=os.path.expanduser("~/datasets/VOC"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super().__init__(root, split, mode, transform, target_transform, **kwargs)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, "SegmentationClass")
        _image_dir = os.path.join(_voc_root, "JPEGImages")
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, "ImageSets/Segmentation")
        if self.mode == "train":
            _split_f = os.path.join(_splits_dir, "train.txt")
        elif self.mode == "val":
            _split_f = os.path.join(_splits_dir, "val.txt")
        elif self.mode == "test":
            _split_f = os.path.join(_splits_dir, "test.txt")
        else:
            raise RuntimeError("Unknown dataset split.")
        self.images = []
        self.masks = []
        # Load all images into ram!!.
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip("\n") + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != "test":
                    _mask = os.path.join(_mask_dir, line.rstrip("\n") + ".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != "test":
            assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.mode == "test":
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == "train":
            img, target = self._sync_transform(img, target)
        elif self.mode == "val":
            img, target = self._val_sync_transform(img, target)
        else:
            assert self.mode == "testval"
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _mask_transform(self, mask):
        target = np.array(mask).astype("int32")
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


class VOCAugSegmentation(BaseDataset):
    voc = [
        "background",
        "airplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorcycle",
        "person",
        "potted-plant",
        "sheep",
        "sofa",
        "train",
        "tv",
    ]
    NUM_CLASS = 21
    TRAIN_BASE_DIR = "VOCaug/dataset/"

    def __init__(
        self,
        root=os.path.expanduser("~/datasets/VOC"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        super().__init__(root, split, mode, transform, target_transform, **kwargs)
        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.TRAIN_BASE_DIR)
        _mask_dir = os.path.join(_voc_root, "cls")
        _image_dir = os.path.join(_voc_root, "img")
        if self.mode == "train":
            _split_f = os.path.join(_voc_root, "train.txt")
        elif self.mode == "val":
            _split_f = os.path.join(_voc_root, "val.txt")
        elif self.mode == "trainval":
            _split_f = os.path.join(_voc_root, "trainval.txt")
        else:
            raise RuntimeError("Unknown dataset split.")
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip("\n") + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != "test":
                    _mask = os.path.join(_mask_dir, line.rstrip("\n") + ".mat")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        if self.mode == "test":
            if self.transform is not None:
                _img = self.transform(_img)
            return _img, os.path.basename(self.images[index])
        _target = self._load_mat(self.masks[index])
        # synchrosized transform
        if self.mode == "train":
            _img, _target = self._sync_transform(_img, _target)
        elif self.mode == "val":
            _img, _target = self._val_sync_transform(_img, _target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _target

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(
            filename, mat_dtype=True, squeeze_me=True, struct_as_record=False
        )
        mask = mat["GTcls"].Segmentation
        return Image.fromarray(mask)

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    train_dataset = VOCSegmentation()
    x_train, y_train = next(iter(train_dataset))
    print(x_train)
    x_train.show()
    print(y_train)

    trainaug_dataset = VOCAugSegmentation()
    x_train, y_train = next(iter(trainaug_dataset))
    print(x_train)
    x_train.show()
    print(y_train)
