#!/usr/bin/env python3
"""Input/Output related functions.
@author: Ninnart Fuengfusin
"""
import logging
import os
from pathlib import Path
from termcolor import colored

__all__ = [
    "G",
    "R",
    "Y",
    "B",
    "gprint",
    "rprint",
    "yprint",
    "bprint",
    "set_logger",
]

G = lambda x: colored(x, "green")
R = lambda x: colored(x, "red")
Y = lambda x: colored(x, "yellow")
B = lambda x: colored(x, "blue")

gprint = lambda x: print(colored(x, "green"))
rprint = lambda x: print(colored(x, "red"))
yprint = lambda x: print(colored(x, "yellow"))
bprint = lambda x: print(colored(x, "blue"))


def set_logger(
    log_path: str,
    log_level: int = logging.INFO,
    to_stdout: bool = True,
    rm_exist: bool = True,
) -> None:
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal
    is saved in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    >>> set_logger('info.log')
    >>> logging.info("Starting training...")
    Args:
        log_path: (string) location of log file.
        log_level: (string) set log level.
        to_stdout: (bool) whether to print log to stdio.
        rm_exist: (bool) remove the old log file before start log or not.
        verbose: (bool) if True, verbose some information.
    """
    assert isinstance(to_stdout, bool)
    assert log_level in [0, 10, 20, 30, 40, 50]
    assert isinstance(rm_exist, bool)

    log_path = os.path.expanduser(log_path)
    path = Path(log_path)
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent, exist_ok=True)

    if rm_exist and os.path.isfile(log_path):
        os.remove(log_path)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s:%(levelname)s:%(filename)s: %(message)s"
            )
        )
        logger.addHandler(file_handler)

        if to_stdout:
            # If True, display logging to a console.
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s:%(levelname)s:%(filename)s: %(message)s"
                )
            )
            logger.addHandler(stream_handler)
