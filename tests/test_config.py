import os
import configparser

from ninpy.config import triplequote2file, autocast


def test_autocast():
    assert type(autocast("one")) == str
    assert type(autocast("5")) == int
    assert type(autocast("1.2345")) == float
    assert type(autocast("1/3")) == str


def test_triplequote2file():
    data = """
    [config]
    one=one
    two=5
    three=1.2345
    four=1/3
    """
    testconfig = os.path.join(os.getcwd(), "testconfig.txt")
    triplequote2file(data, testconfig)
    parser = configparser.ConfigParser()
    parser.read(testconfig)

    # Success or fail test remove file first.
    os.remove(testconfig)
    assert parser.get("config", "one") == "one"
    assert parser.get("config", "two") == "5"
    assert parser.get("config", "three") == "1.2345"
    assert parser.get("config", "four") == "1/3"
