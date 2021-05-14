import numpy as np
from albumentations.pytorch import ToTensorV2

from ninpy.datasets import WrappedCompose


def test_wrapped_compose():
    compose = WrappedCompose([ToTensorV2()])
    output = compose(np.zeros((32, 32, 3)))
    assert True
