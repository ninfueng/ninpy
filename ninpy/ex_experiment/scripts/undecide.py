#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 20:27:21 2021

@author: ninnart
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import Counter
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, vgg16_bn, alexnet, resnet18
from torch.nn.utils import prune
import urllib
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import glob
import time
import json
import logging
import os
import shutil
import torch



def set_batchnorm_eval(m) -> None:
    """From: https://discuss.pytorch.org/t/cannot-freeze-batch-normalization-parameters/38696
    Ex:
        model.apply(set_batchnorm_eval)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_batchnorm(m) -> None:
    """
    Ex:
        model.apply(freeze_batchnorm)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for param in m.parameters():
            param.requires_grad = False
            

def freeze_param_given_name(model, freeze_list: list, verbose: bool = False) -> None:
    """
    """
    for name, param in model.named_parameters():
        if name in freeze_list:
            param.requires_grad = False
            if verbose:
                logging.info(f'{name} was freeze.')


def convert_module2module(
        model, module_a, module_b, 
        verbose: bool = True) -> None:
    """From: https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/5
    Ex:
        convert_module2module(model, HardSwish, nn.Hardswish())
    """
    assert isinstance(verbose, bool)
    for child_name, child in model.named_children():
        if isinstance(child, module_a):
            setattr(model, child_name, module_b)
            if verbose:
                print(f'Replace {module_a} to {module_b}')
        else:
            convert_module2module(child, module_a, module_b)
            


def test_get_imagenet_img():
    """Test model for `get_imagenet_img` function.
    """
    test_img = get_imagenet_img()
    model = alexnet(pretrained=True)
    out = model(test_img)[0]
    pred_label = out.softmax(0).argmax()
    assert pred_label == 258


def get_all_submodule(model: nn.Module):
    """Recursivly get list of all children.
    Ex:
        model = vgg16(pretrained=False)
        print(get_all_children(model))
    TODO: 
        Not work in resnet18 len 14 != 52.
        Find way to 
    """
    isinstance(model, nn.Module)
    list_children = []
    for c in model.children():
        if not isinstance(c, nn.Sequential):
            list_children += [c]
        else:
            list_children += get_all_submodule(c)
    return list_children


def test_get_all_submodule():
    # model2 = resnet18(pretrained=False)
    # sub_modules2 = get_all_submodule(model2)
    # assert len(sub_modules2) == 39
    model = vgg16(pretrained=False)
    sub_modules = get_all_submodule(model)
    assert len(sub_modules) == 39
        



def normal_init(m):
    """From: https://github.com/pytorch/examples/blob/master/dcgan/main.py
    Using model.apply(normal_init)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

