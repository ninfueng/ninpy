#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
import urllib

import torch
from PIL import Image
from torchvision import transforms


def get_imagenet_image(preprocess: bool = False) -> torch.Tensor:
    """From: https://pytorch.org/hub/pytorch_vision_alexnet/
    https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    Correct label should be 258 or Samoyed, Samoyede.
    Download an imagenet image `dog.jpg` from Pytorch repository.
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
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_image = preprocess(input_image).unsqueeze(0)
    os.remove(filename)
    return input_image
