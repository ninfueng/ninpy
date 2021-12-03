import argparse

from ninpy.experiment import args2str


def test_argsstr():
    parser = argparse.ArgumentParser(description="Test config.")
    parser.add_argument("--var0", type=int, default=1)
    parser.add_argument("--var1", type=int, default=2)

    # Fixes error from pytest.
    # From: https://stackoverflow.com/questions/55259371/pytest-testing-parser-error-unrecognised-arguments
    args = parser.parse_args([])
    argsstr = args2str(args)
    assert argsstr == "var0:1-var1:2-"
