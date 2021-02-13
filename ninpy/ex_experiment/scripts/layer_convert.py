#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:34:30 2021

@author: ninnart
"""


class LayerConverter(object):
    """Collection of converter for layer to another type of layer.
    Modified from: From: https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104686
    """
    @staticmethod
    def convert_bn_layers(model, old_layer_type, new_layer_type, convert_weights=False, num_groups=None):
        """If num_groups is 1, GroupNorm turns into LayerNorm. If num_groups is None, GroupNorm turns into InstanceNorm
        Ex:
            LayerConverter.convert_bn_layers(model, nn.BatchNorm2d, nn.GroupNorm, True, num_groups=2)
        """
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.convert_bn_layers(module, old_layer_type, new_layer_type, convert_weights, num_groups=num_groups)
            # single module
            if type(module) == old_layer_type:
                old_layer = module
                new_layer = new_layer_type(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 
                if convert_weights:
                    new_layer.weight = old_layer.weight
                    new_layer.bias = old_layer.bias
                model._modules[name] = new_layer
        return model

    @staticmethod
    def convert_conv_layers(model, old_layer_type, new_layer_type, convert_weights=False, **kwargs):
        """
        Example:
        ```
        from torchvision.models import vgg16
        m = vgg16(pretrained=False)
        LayerConverter.convert_conv_layers(m, nn.Conv2d, nn.Conv1d)
        ```
        """
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.convert_conv_layers(module, old_layer_type, new_layer_type, convert_weights, **kwargs)
            # single module
            if type(module) == old_layer_type:
                old_layer = module            
                if module.bias is None:
                    bias = False
                else:
                    bias = True
                new_layer = new_layer_type(
                    in_channels=module.in_channels, out_channels=module.out_channels, 
                    kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                    dilation=module.dilation, groups=module.groups, bias=bias, **kwargs)
                if convert_weights:
                    new_layer.weight = old_layer.weight
                    new_layer.bias = old_layer.bias
                model._modules[name] = new_layer
        return model

    @staticmethod
    def convert_activation(model, old_layer_type, new_layer_type, **kwargs):
        """
        Example:
        ```
        from torchvision.models import vgg16
        m = vgg16(pretrained=False)
        LayerConverter.convert_conv_layers(m, nn.ReLU, nn.ReLU6)
        ```
        """
        from torch.nn.modules.module import ModuleAttributeError
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recurives.
                model._modules[name] = LayerConverter.convert_activation(module, old_layer_type, new_layer_type, **kwargs)
            # single module
            if type(module) == old_layer_type:
                try:
                    new_layer = new_layer_type(inplace=module.inplace)
                except ModuleAttributeError:
                    # Activation without inplace attribute.
                    # Problem with thrid party built activation without the inplace as the input. 
                    new_layer = new_layer_type()
                model._modules[name] = new_layer
        return model