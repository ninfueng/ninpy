"""For new base module"""
from typing import Callable

import torch.nn as nn


class DataParallel(nn.DataParallel):
    """Allows DataParallel to access attributes.
    Modified: https://github.com/pytorch/pytorch/issues/16885"""

    def __getattr__(self, name: str) -> Callable:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
