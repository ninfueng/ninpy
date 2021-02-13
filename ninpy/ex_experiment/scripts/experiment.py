#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:27:19 2021

@author: ninnart
"""
import os
import time
import glob
import shutil
import logging
from functools import reduce


def set_experiment(experiment_path: str, match_list = ['*.py', '*.sh', '*.yaml'],  verbose: bool = False) -> str:
    """Inspired from: https://github.com/VITA-Group/FasterSeg/blob/master/tools/utils/darts_utils.py
    Create a folder with the name f'{datetime}-{experiment_path}' format.
    With scripts folder into it and copy all scripts within `list_types` into this folder. 
    Example:
        set_experiment(experiment_path)
    """
    assert isinstance(experiment_path, str)
    assert len(match_list) > 0
    assert isinstance(verbose, bool)
    datetime = time.strftime("%Y:%m:%d-%H:%M:%S")
    path = f'{datetime}-{experiment_path}'
    
    if not os.path.exists(path):
        os.mkdir(path)
        #os.makedirs(os.path.join(path, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        if verbose:
            logging.info(f'Mkdir at {path}.')
            
    save_scripts = [glob.glob(t, recursive=False) for t in match_list]
    save_scripts = reduce(lambda x, y: x + y, save_scripts)
    
    for s in save_scripts:
        dst_path = os.path.join(path, 'scripts', str(s)) 
        shutil.copyfile(s, dst_path)
        if verbose:
            logging.info(f'Copy {s} to {dst_path}.')
    return path


if __name__ == '__main__':
    set_experiment('experiment', ['*.py', '*yaml', 'test/*.py'])