#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:04:34 2021

@author: ninnart
"""
import os
import logging
from pathlib import Path


def set_logger(log_path: str, to_console: bool = True) -> None:
    """From: https://github.com/cs230-stanford/cs230-code-examples
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    assert isinstance(to_console, bool)
    path = Path(log_path)
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(filename)s: %(message)s'))
        logger.addHandler(file_handler)

        if to_console:
            # Logging to console
            # This allows tqdm only to process in console.
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(filename)s: %(message)s'))
            logger.addHandler(stream_handler)


if __name__ == '__main__':
    set_logger('test/test.log', False)
    logging.info('test')
