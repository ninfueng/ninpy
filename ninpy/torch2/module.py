"""For new base module"""
from typing import Callable

import torch.nn as nn


class DataParallel(nn.DataParallel):
    """This DataParallel allows to access any attributes from the module.
    This class is not necessary for a newer version of DataParallel.
    """

    def __getattr__(self, name: str) -> Callable:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
