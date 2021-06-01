#!/usr/bin/env python3
"""A collection of data structure functions.
@author: Ninnart Fuengfusin
"""
import logging
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch


class AttrDict(OrderedDict):
    """Recusive attribute OrderedDict.
    Example:
    >>> attrdict = AttrDict(
        {"test": {"test2": 1}, "recursive": [1, 2, 3, {"test3": {"test4": 4}}]})
    >>> attrdict.test
    AttrDict([('test2', 1)])
    >>> attrdict.recursive[-1].test3.test4
    4
    """

    __slots__ = ()
    __getattr__ = OrderedDict.__getitem__
    __setattr__ = OrderedDict.__setitem__

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        for k in self.keys():
            if isinstance(self[k], dict):
                self[k] = AttrDict(self[k])
            elif isinstance(self[k], list):
                for idx, v in enumerate(self[k]):
                    if isinstance(v, dict):
                        self[k][idx] = AttrDict(self[k][idx])

    def to_dict(self) -> dict:
        """Convert to a dict """
        for k in self.keys():
            if isinstance(self[k], list) or isinstance(self[k], tuple):
                # Not support standard datatypes.
                self[k] = torch.as_tensor(self[k])
        return dict(self)


class AttrDictList(AttrDict):
    """AttrDict with initialization with the lists.
    With additional methods to support.
    Example:
    >>> dictlist = AttrDictList('book0', 'book1')
    >>> dictlist.book0.append(5)
    >>> dictlist.book0.append(10)
    >>> dictlist.to_csv('book.csv', 0)
    """

    def __init__(self, *args) -> None:
        [self.update({arg: []}) for arg in args]

    def _equal_len(self, filled: int = 0) -> dict:
        """Make all lists in dict have the same len."""
        # Get maxlen from all keys.
        maxlen = -sys.maxsize - 1
        for k in self.keys():
            if len(self[k]) > maxlen:
                maxlen = len(self[k])

        # Fill list in dictlist that len less than maxlen to maxlen.
        for k in self.keys():
            if len(self[k]) < maxlen:
                diff = maxlen - len(self[k])
                _ = [self[k].append(filled) for _ in range(diff)]

    def to_df(self, filled: float = np.nan) -> pd.DataFrame:
        """Convert to DataFrame required to filling all missing values
        in case the lengths of list are not equal.
        """
        self._equal_len(filled=filled)
        df = pd.DataFrame(self)
        return df

    def to_csv(
        self, file_name: str, filled: float = np.nan, verbose: str = True
    ) -> None:
        """Saving the dictlist to csv."""
        assert isinstance(file_name, str)
        df = self.to_df(filled=filled)
        df.to_csv(file_name, index=None)
        if verbose:
            logging.info(f"Save csv@{file_name}.")

    def append_kwargs(self, **kwargs) -> None:
        """Append multiple lists in dict at the same time using kwargs."""
        _ = [self[key].append(kwargs[key]) for key in kwargs.keys()]

    def to_dict(self) -> dict:
        for k in self.keys():
            if isinstance(self[k], list) or isinstance(self[k], tuple):
                # Not support standard datatypes.
                self[k] = torch.as_tensor(self[k])
        return dict(self)

    def to_tensorboard(self) -> None:
        # TODO: support a tensorboard. Expected idx as epoch.
        # Using idx of list with add scalar with for loop.
        raise NotImplementedError("Not supported yet.")

    def to_yaml() -> None:
        # TODO: support dict to yaml.
        raise NotImplementedError("Not supported yet.")
