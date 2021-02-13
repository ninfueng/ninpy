#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:06:20 2021

@author: ninnart
"""


def convert_float_int8(x: float, scale: float, zero_point: int) -> int:
    """Simulate converting int8 qunatization.
    Ex:
        convert_float_int8(0.5841, 0.02987, 50)
        -> 70
    """
    x_int8 = (x/scale) + zero_point
    return round(x_int8)


def convert_int8_float(x: int, scale: float, zero_point: int) -> float:
    """Simulate converting int8 qunatization.
    Without rounding in same case as `convert_float_int8`.
    Ex:
        convert_int8_float(70, 0.02987, 50)
        -> 0.5974
    """
    x_float = (x - zero_point)*scale
    return x_float

