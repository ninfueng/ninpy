import os
from typing import Callable, Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from ninpy.torch2.datasets.augment import IMAGENET_MEAN, IMAGENET_STD
from torch.utils.data.dataset import Dataset


# TODO: VOCAugSegmentationDataset BASE_AUG = 'benchmark_RELEASE/dataset'
# TODO: Concate two different Dataset.
class VOCSegmentationDataset(Dataset):
    """TODO: downloading dataset and aug dataset.
    Modified:
        https://albumentations.ai/docs/autoalbument/examples/pascal_voc/
        https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/scripts/prepare_pascal.py
    """

    # TODO: remove these two lists into a dict.
    VOC_CLASSES = [
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

    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]

    def __init__(
        self, root: str, train: bool, transform: Optional[Callable]
    ) -> None:
        assert isinstance(root, str)
        assert isinstance(train, bool)

        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        BASE_VOC = "VOCdevkit/VOC2012/"
        base_dir = os.path.join(self.root, BASE_VOC)
        mask_dir = os.path.join(base_dir, "SegmentationClass")
        image_dir = os.path.join(base_dir, "JPEGImages")
        label_dir = os.path.join(base_dir, "ImageSets/Segmentation")

        if train:
            label_txt = os.path.join(label_dir, "train.txt")
        else:
            label_txt = os.path.join(label_dir, "val.txt")

        labels = np.loadtxt(label_txt, dtype=str).tolist()
        self.images = [os.path.join(image_dir, l + ".jpg") for l in labels]
        if train:
            self.masks = [os.path.join(mask_dir, l + ".png") for l in labels]
            assert len(self.images) == len(self.masks)
        else:
            self.mask = None

    def __getitem__(self, index: int):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            mask = cv2.imread(self.masks[index], cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = self.cvt2mask(mask)

        if self.transform is not None:
            if self.train:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                return image, mask
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]
                return image
        else:
            if self.train:
                # TODO: same poicy as the imagenet, therefore using it instead.
                transform = A.Compose(
                    [
                        A.RandomResizedCrop(480, 480),
                        A.HorizontalFlip(),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
                transformed = transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                return image, mask
            else:
                transform = A.Compose(
                    [
                        A.Resize(520, 520),
                        A.CenterCrop(480, 480),
                        A.HorizontalFlip(),
                        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                        ToTensorV2(),
                    ]
                )
                transformed = transform(image=image, mask=mask)
                image = transformed["image"]
                return image

    def cvt2mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert a mask to class array."""
        height, width = mask.shape[:2]
        colormap = self.VOC_COLORMAP
        segmentation_mask = np.zeros(
            (height, width, len(colormap)), dtype=np.float32
        )

        for label_index, label in enumerate(colormap):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)
        return segmentation_mask


if __name__ == "__main__":
    # from ninpy.debug import show_torch_image

    # # train_loader, val_loader = get_voc2012_loader(
    # #     '/home/ninnart/datasets/VOC',
    # #     False, None, 128, 8, False, None, None)

    # # test_batch = next(iter(train_loader))
    # # print(test_batch)

    # train_transforms = A.Compose(
    #     [
    #         A.RandomResizedCrop(480, 480),
    #         A.HorizontalFlip(),
    #         A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    #         ToTensorV2(),
    #     ]
    # )

    # root = "~/datasets/VOC"
    # root = os.path.expanduser(root)
    # train_dataset = VOCSegmentationDataset(
    #     root=root, train=True, transform=train_transforms
    # )
    # img, mask = next(iter(train_dataset))
    # show_torch_image(img, True)
    VOC_CLASSES = [
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

    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
