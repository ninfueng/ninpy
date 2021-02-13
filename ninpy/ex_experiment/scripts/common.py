#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:32:45 2021

@author: ninnart
"""
import torch
import numpy as np
import logging


def save_model(
    save_dir: str, model, optimizer, 
    metric = None, epoch: int = None, 
    verbose: bool = True) -> None:
    r"""
    """
    torch.save({
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'metric': metric,
        'epoch': epoch
        },
        save_dir)
    if verbose:
        logging.info(f'Save model@ {save_dir}')


def load_model(save_dir: str, model, optimizer, verbose: bool = False):
    """
    """
    ckpt = torch.load(save_dir)
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    metric = ckpt['metric']
    epoch = ckpt['epoch']
    model = model.load_state_dict(model_state_dict)
    optimizer = optimizer.load_state_dict(optimizer_state_dict)
    if verbose:
        logging.info(f'Load a model with score {metric}@ {epoch} epoch')
    return  model, optimizer


def add_weight_decay(model, weight_decay, skip_list=()) -> None:
    """From: https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/3
    https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/
    Ex:
        
    """
    assert isinstance(weight_decay, float)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue # frozen weights		            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list: 
            no_decay.append(param)
        else: 
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]



def make_onehot(input, num_classes: int):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    assert isinstance(num_classes, int)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).to(input.device)
    result = result.scatter_(1, input, 1)
    return result



class RunningAverage(object):
    """From: https://github.com/cs230-stanford/cs230-code-examples
    A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)