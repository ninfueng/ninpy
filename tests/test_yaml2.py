import argparse
import os

from ninpy.yaml2 import args2str, dict2str, dump_yaml, load_yaml


def test_dump_yaml():
    LOAD_CONFIG = "test_dumped.yaml"
    test_dict = {1: 2, 3: 4, 5: {6: 7}, 8: {9: {10: 11}}}

    dump_yaml(test_dict, LOAD_CONFIG)
    loaded_dict = load_yaml(LOAD_CONFIG)
    os.remove(LOAD_CONFIG)

    assert test_dict == loaded_dict
    assert dict2str(test_dict) == "1:2-3:4-5::6:7-8::9::10:11-"


def test_argsstr():
    parser = argparse.ArgumentParser(description="Test config.")
    parser.add_argument("--var0", type=int, default=1)
    parser.add_argument("--var1", type=int, default=2)
    args = parser.parse_args()

    argsstr = args2str(args)
    assert argsstr == "var0:1-var1:2-"
