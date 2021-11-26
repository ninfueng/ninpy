import os
import argparse

from ninpy.yaml2 import dump_yaml, load_yaml
from ninpy.experiment import dict2str


def test_dump_yaml():
    LOAD_CONFIG = "test_dumped.yaml"
    test_dict = {1: 2, 3: 4, 5: {6: 7}, 8: {9: {10: 11}}}

    dump_yaml(test_dict, LOAD_CONFIG)
    loaded_dict = load_yaml(LOAD_CONFIG)
    os.remove(LOAD_CONFIG)

    assert test_dict == loaded_dict
    assert dict2str(test_dict) == "1:2-3:4-5::6:7-8::9::10:11-"
