#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:18:09 2021

@author: ninnart
"""
import sys
from collections import OrderedDict
import numpy as np
import pandas as pd
import logging


class AttributeDict(dict):
    """From: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    Example:
    ```
    a = AttributeDict({'d': 2})
    a.d
    > 2
    a.test = 5
    a
    > {'d': 2, 'test': 5}
    ```
    """
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class AttributeOrderedDict(OrderedDict):
    """From: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    Same case as `AttributeDict`, however with support from OrderedDict instead.
    Example:
    ```
    a = AttributeOrderedDict({'d': 2})
    a.d
    > 2
    a.test = 5
    a
    > {'d': 2, 'test': 5}
    ```
    """
    __slots__ = () 
    __getattr__ = OrderedDict.__getitem__
    __setattr__ = OrderedDict.__setitem__


class AttributeOrderedDictList(AttributeOrderedDict):
    """defaultdict with list might in sameway as this.
    With additional methods to support.
    Example:
    ```
    dictlist = AttributeOrderedDictList('book0', 'book1')    
    dictlist.book0.append(5)
    dictlist.book0.append(10)
    dictlist.to_csv('book.csv', 0)
    ```
    """
    def __init__(self, *args):
        for arg in args:
            self.update({arg: []})
            
    def fill_dictlist_equal_len(self, fill_var: int = 0) -> dict:
        """Make all lists in dict have the same len.
        """
        # Get maxlen from all keys.
        maxlen = -sys.maxsize - 1
        for k in self.keys():
            if len(self[k]) > maxlen:
                maxlen = len(self[k])

        # Fill list in dictlist that len less than maxlen to maxlen.
        for k in self.keys():
            if len(self[k]) < maxlen:
                diff =  maxlen - len(self[k])
                _ = [self[k].append(fill_var) for i in range(diff)]

    def to_df(self, fill_var: float = np.nan) -> pd.DataFrame:
        """
        """
        self.fill_dictlist_equal_len(fill_var=fill_var)
        df = pd.DataFrame(self)
        return df
    
    def to_csv(self, file_name: str, fill_var: float = np.nan) -> None:
        """Saving to csv.
        """
        df = self.to_df(fill_var=fill_var)
        df.to_csv(file_name, index=None)
        logging.info(f'Save csv@{file_name}.')

    def append_kwargs(self, **kwargs) -> None:
        """Append multiple lists in dict at the same time using kwargs.
        """
        _ = [self[key].append(kwargs[key]) for key in kwargs.keys()]


if __name__ == '__main__':
    dictlist = AttributeOrderedDictList('book0', 'book1')    
    dictlist.book0.append(5)
    dictlist.book0.append(10)
    dictlist.to_csv('book.csv', 0)