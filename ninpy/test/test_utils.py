import pytest
from torchvision.models import alexnet
from ninpy.debug import get_imagenet_img


def test_get_imagenet_img():
    """Only checking for working or not.
    """
    test_img = get_imagenet_img()
    model = alexnet(pretrained=False)
    out = model(test_img)[0]
    # pred_label = out.softmax(0).argmax()
    # assert pred_label == 258
    assert out is not None
