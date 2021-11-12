#!/usr/bin/env python3
"""IO related functions.
@author: Ninnart Fuengfusin
"""
import logging
import os
from pathlib import Path

from termcolor import colored

gprint = lambda x: print(colored(x, "green"))
rprint = lambda x: print(colored(x, "red"))
yprint = lambda x: print(colored(x, "yellow"))
bprint = lambda x: print(colored(x, "blue"))


def set_logger(
    log_pth: str,
    to_console: bool = True,
    rm_exist: bool = True,
    verbose: bool = True,
) -> None:
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    >>> set_logger('info.log')
    >>> logging.info("Starting training...")
    Args:
        log_pth: (string) where to log.
        to_console: (bool) show log in a console or not.
        rm_exist: (bool) remove the old log file before start log or not.
        verbose: (bool) if True, verbose some information.
    """
    assert isinstance(to_console, bool)
    assert isinstance(rm_exist, bool)
    assert isinstance(verbose, bool)

    log_pth = os.path.expanduser(log_pth)
    path = Path(log_pth)
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent, exist_ok=True)
    if rm_exist and os.path.isfile(log_pth):
        os.remove(log_pth)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_pth)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(filename)s: %(message)s")
        )
        logger.addHandler(file_handler)

        if to_console:
            # If True, display logging to a console.
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(filename)s: %(message)s")
            )
            logger.addHandler(stream_handler)

    if verbose:
        logging.info(f"Log file at a location: {log_pth}.")
