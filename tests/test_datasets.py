import os

import numpy as np
from albumentations.pytorch import ToTensorV2

from ninpy.torch2.datasets import BurstImageFolder
from ninpy.torch2.datasets.augment import ClassifyCompose


def test_wrapped_compose():
    compose = ClassifyCompose([ToTensorV2()])
    output = compose(np.zeros((32, 32, 3)))
    assert output.shape == (3, 32, 32)


def test_burstimagefolder():
    curdir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(curdir, "images")

    dataset = BurstImageFolder(img_dir)
    dataset.load_images()
    img, label = next(iter(dataset))

    assert img.size == (512, 512)
    assert label == 0
