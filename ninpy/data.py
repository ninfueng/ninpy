#!/usr/bin/env python3
"""A collection of data structure functions.
@author: ninfueng
"""
import json
import sys
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from ninpy.config import dump_yaml

__all__ = ["AttrDict", "AttrDictList"]


class AttrDict(OrderedDict):
    """OrderedDict that can accessed attributes as a method.
    This allows `.attributes` instead of `['attributes']`.
    Example:
    >>> attrdict = AttrDict({"test": {"test2": 1}, "recursive": [1, 2, 3, {"test3": {"test4": 4}}]})
    >>> attrdict.test
    AttrDict([('test2', 1)])
    >>> attrdict.recursive[-1].test3.test4
    4
    """

    __slots__ = ()
    __getattr__ = OrderedDict.__getitem__
    __setattr__ = OrderedDict.__setitem__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for k in self.keys():
            if isinstance(self[k], dict):
                # Recursive.
                self[k] = AttrDict(self[k])
            elif isinstance(self[k], list):
                for idx, v in enumerate(self[k]):
                    # Detect any dict in the list.
                    if isinstance(v, dict):
                        self[k][idx] = AttrDict(self[k][idx])

    def to_yaml(self, filename: str) -> None:
        assert isinstance(filename, str)
        dump_yaml(self, filename)

    def to_json(self, filename: str, indent: int = 4) -> None:
        """Save data in AttrDict to a json file."""
        assert isinstance(filename, str)
        assert isinstance(indent, int)
        with open(filename, "w") as f:
            json.dump(self, f, indent=indent)


class AttrDictList(AttrDict):
    """AttrDict with initialization with the lists.
    With additional methods to support.
    Example:
    >>> dictlist = AttrDictList('book0', 'book1')
    >>> dictlist.book0.append(5)
    >>> dictlist.book0.append(10)
    >>> dictlist.to_csv('book.csv', 0)
    """

    def __init__(self, *args: str, **kwargs: Any) -> None:
        [self.update({arg: []}) for arg in args]
        assert all([isinstance(v, list) for _, v in kwargs])
        [self.update({k: v}) for k, v in kwargs]

    def _cvt2list(self) -> None:
        """If one of values in `AttrDictList` is another type  than the `list`.
        Attempt to convert to `list` before."""
        for k in self.keys():
            value = self[k]
            if hasattr(value, "shape"):
                if len(value.shape) > 1:
                    raise NotImplementedError(
                        f"Detect value in {k} key is a multi-dimensional array."
                    )
            if isinstance(value, dict):
                raise NotImplementedError(f"Detect value in {k} key is a dict.")
            elif hasattr(value, "tolist"):
                # Support `torch.Tensor`, `np.ndarray`, and `pd.Series`.
                self[k] = value.tolist()
            elif hasattr(value, "numpy"):
                # Support `tf.constant`.
                self[k] = value.numpy().tolist()
            else:
                warnings.warn(f"Did not convert value {value} in {k} keys.")

    def _fill_equal_len(self, fill: int = 0) -> None:
        """Fill all values to a same len. If the value is not list,"""
        self._cvt2list()

        maxlen = -sys.maxsize - 1
        for k in self.keys():
            len_ = len(self[k])
            if len_ > maxlen:
                maxlen = len_

        for k in self.keys():
            len_ = len(self[k])
            if len_ < maxlen:
                diff = maxlen - len_
                [self[k].append(fill) for _ in range(diff)]

    def to_df(self, fill: Any = np.nan) -> pd.DataFrame:
        """Convert to DataFrame required to filling all missing values
        in case the lengths of list are not equal.
        """
        self._fill_equal_len(fill=fill)
        df = pd.DataFrame(self)
        return df

    def to_csv(self, name: str, fill: Any = np.nan) -> None:
        df = self.to_df(fill=fill)
        df.to_csv(name, index=False)

    def append_kwargs(self, **kwargs: Any) -> None:
        """Append multiple lists in dict at the same time using kwargs."""
        [self[key].append(kwargs[key]) for key in kwargs.keys()]
