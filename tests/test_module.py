#!/usr/bin/env python3
import torch.nn as nn

from ninpy.torch2 import DataParallel


def test_dataparallel():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()

        def test_case(self):
            return True

    base = Base()
    out = base.test_case()
    baseparallel = DataParallel(base)
    out0 = baseparallel.test_case()
    assert out == out0
