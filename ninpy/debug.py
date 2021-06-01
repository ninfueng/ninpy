#!/usr/bin/env python3
"""@author: Ninnart Fuengfusin"""
import os
import urllib

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from ninpy.datasets.augment import get_imagenet_transforms


def get_imagenet_image(preprocess: bool = False) -> torch.Tensor:
    """Correct label should be `258` or `Samoyed, Samoyede`.
    Download an imagenet image `dog.jpg` from PyTorch repository.
    Transform the image into PyTorch format and ready to process in Imagenet trained models.
    """
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    if preprocess:
        preprocess_fn = get_imagenet_transforms(256, 224)[1]
        input_image = preprocess_fn(input_image).unsqueeze(0)
    os.remove(filename)
    return input_image


def show_torch_image(x: torch.Tensor, denormalize: bool = False) -> None:
    """Show an image from torch format with an option to denormalize ImageNet normalized image.
    For example:
    >>> show_image(torch.zeros(3, 224, 224), False)
    """
    assert isinstance(x, torch.Tensor)
    assert isinstance(denormalize, bool)
    assert len(x.shape) == 3, f"Expect shape of x is 3, Your: {x.shape}"

    if denormalize:
        inverse_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
        )
        x = inverse_normalize(x)
    x = x.transpose(0, 2).detach().cpu().numpy()
    plt.imshow(x)
    plt.show()
