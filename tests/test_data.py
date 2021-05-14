import os

import pandas as pd

from ninpy.data import AttrDict, AttributeOrderedDictList


def test_attribute_ordered_dictlist():
    """"""
    dictlist = AttributeOrderedDictList("book0", "book1")
    dictlist.book0.append(5)
    dictlist.book0.append(10)

    dictlist.to_csv("book.csv", 0)
    df = pd.read_csv("book.csv")
    os.remove("book.csv")

    assert df["book0"][0] == 5
    assert df["book0"][1] == 10
    assert df["book1"][0] == 0


def test_attrdict():
    """Recursive testing of AttrDict."""
    attrdict = AttrDict(
        {"test": {"test2": 1}, "recursive": [1, 2, 3, {"test3": {"test4": 4}}]}
    )
    assert attrdict.test.test2 == 1
    assert attrdict.recursive[-1].test3.test4 == 4
