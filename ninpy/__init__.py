import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(filename)s: %(message)s")
)
logger.addHandler(stream_handler)

from ninpy import config, experiment, job, resize, torch2
from ninpy.common import *
from ninpy.data import *
from ninpy.log import *
from ninpy.version import *
