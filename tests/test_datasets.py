import numpy as np
from albumentations.pytorch import ToTensorV2

from ninpy.datasets.augment import ClassifyCompose


def test_wrapped_compose():
    compose = ClassifyCompose([ToTensorV2()])
    output = compose(np.zeros((32, 32, 3)))
    assert output.shape == (3, 32, 32)
