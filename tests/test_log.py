import logging
import os

from ninpy.log import set_logger


def test_set_logger():
    set_logger("./test_set_logger/test.log", False)
    logging.info("This is a testing log.")
    logging.info("This is a testing log2.")
    logging.shutdown()
    # assert os.path.isfile("./test_set_logger/test.log")
    # os.remove("./test_set_logger/test.log")
    # os.rmdir("./test_set_logger")
