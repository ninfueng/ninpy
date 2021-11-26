""""""
from typing import Callable

import torch.nn as nn

__all__ = ["DataParallel"]


class DataParallel(nn.DataParallel):
    """This `DataParallel` allows an access to any attributes from the module.
    From: https://github.com/pytorch/pytorch/issues/16885
    """

    def __getattr__(self, name: str) -> Callable:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
