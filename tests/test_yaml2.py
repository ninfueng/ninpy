import argparse
import os

from ninpy.config import dump_json, dump_yaml, json2argparse, load_yaml
from ninpy.experiment import dict2str


def test_dump_yaml() -> None:
    LOAD_CONFIG = "./test_dumped.yaml"
    test_dict = {1: 2, 3: 4, 5: {6: 7}, 8: {9: {10: 11}}}

    dump_yaml(test_dict, LOAD_CONFIG)
    loaded_dict = load_yaml(LOAD_CONFIG)
    os.remove(LOAD_CONFIG)

    assert test_dict == loaded_dict
    assert dict2str(test_dict) == "1:2-3:4-5::6:7-8::9::10:11-"


def test_dump_json() -> None:
    parser = argparse.ArgumentParser(description="Testing argparse to json.")
    parser.add_argument("--a", type=int, default=123)
    parser.add_argument("--b", type=int, default=456)
    args = parser.parse_args([])

    json_path = "./test.json"
    dict_ = {"a": 1, "b": 4}
    dump_json(dict_, json_path)
    args = json2argparse(json_path, args)

    if os.path.exists(json_path):
        os.remove(json_path)

    assert args.a == 1
    assert args.b == 4
