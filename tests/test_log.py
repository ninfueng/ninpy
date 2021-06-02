import logging
import os

from ninpy.log import set_logger


def test_set_logger():
    curdir = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(curdir, "test_set_logger", "test.log")
    set_logger(logdir, False)
    logging.info("This is a testing log.")
    os.rmdir(os.path.join(curdir, "test_set_logger"))
    #assert os.path.isfile(logdir)
    #os.remove(logdir)
