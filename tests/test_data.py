import os
import pandas as pd

from ninpy.data import AttributeOrderedDictList


def test_attribute_ordered_dictlist():
    dictlist = AttributeOrderedDictList('book0', 'book1')
    dictlist.book0.append(5)
    dictlist.book0.append(10)
    dictlist.to_csv('book.csv', 0)
    df = pd.read_csv('book.csv')
    os.remove('book.csv')

    assert df['book0'][0] == 5
    assert df['book0'][1] == 10
    assert df['book1'][0] == 0