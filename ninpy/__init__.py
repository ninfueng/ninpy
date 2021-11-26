import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(filename)s: %(message)s")
)
logger.addHandler(stream_handler)


from ninpy.common import *
from ninpy.log import *
from ninpy.data import *
from ninpy.version import *
from ninpy import experiment, hyper, job, resize, torch2, yaml2
