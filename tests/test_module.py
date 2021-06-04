#!/usr/bin/env python3
import torch.nn as nn

from ninpy.torch2.module import DataParallel


def test_dataparallel():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()

        def test_case(self):
            return True

    base = Base()
    out = base.test_case()
    baseparallel = DataParallel(base)

    # Can access an attribute or not.
    out0 = baseparallel.test_case()
    assert out == out0
