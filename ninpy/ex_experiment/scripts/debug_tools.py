#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:26:57 2021

@author: ninnart
"""
import urllib
import torch
from torchvision import transforms
from PIL import Image


def get_imagenet_img() -> torch.Tensor:
    """From: https://pytorch.org/hub/pytorch_vision_alexnet/
    https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    Correct label should be 258 or Samoyed, Samoyede.
    Download an imagenet image `dog.jpg` from Pytorch repository.
    Transform the image into Pytorch format and ready to process in Imagenet trained models.
    """
    url, filename = (
        'https://github.com/pytorch/hub/raw/master/images/dog.jpg', 'dog.jpg')
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])
    input_tensor = preprocess(input_image).unsqueeze(0)
    return input_tensor

if __name__ == '__main__':
    from torchvision.models import resnet18

    img = get_imagenet_img()
    model = resnet18(pretrained=False)
    pred = model(img)
    print(pred)