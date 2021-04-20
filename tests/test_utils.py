import shutil
from multiprocessing import cpu_count

import torch
from torchvision.models import alexnet, resnet18

from ninpy.datasets import load_toy_dataset
from ninpy.debug import get_imagenet_img
from ninpy.models import resnet20
from ninpy.torch_utils import get_num_weight_from_name, topk_accuracy


def test_get_imagenet_img():
    r"""Checking for an example imagenet link is dead for not."""
    PRETRAINED = False
    test_img = get_imagenet_img(preprocess=True)
    model = alexnet(pretrained=PRETRAINED)
    out = model(test_img)[0]
    if PRETRAINED:
        pred_label = out.softmax(0).argmax()
        assert pred_label == 258
    assert out is not None


def test_get_num_weight_from_name():
    model = resnet18(pretrained=False)
    num_weight = get_num_weight_from_name(model, "fc")
    assert num_weight == 512 * 1_000

    num_weight = get_num_weight_from_name(model, "layer1.0.conv1")
    assert num_weight == 64 * 64 * 3 * 3
