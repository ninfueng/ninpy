#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:49:50 2021

@author: ninnart
"""

import sys
import os
import logging
from subprocess import check_call
from config import dump_yaml, load_yaml


def launch_job(
        job_name: str,
        script_name: str,
        hyper_params: dict,
        verbose: bool = False) -> None:
    """Modified from: https://github.com/cs230-stanford/cs230-code-examples
    """
    # Launch training with this config
    assert isinstance(hyper_params, dict)
    PYTHON = sys.executable
    # For saving yaml.
    hyper_yaml = f'{job_name}.yaml'
    dump_yaml(hyper_params, hyper_yaml)
    cmd = f'{PYTHON} {script_name} --yaml={hyper_yaml}'
    if verbose:
        logging.info(cmd)
    check_call(cmd, shell=True)
    os.remove(hyper_yaml)


if __name__ == '__main__':
    hyper_params = load_yaml('test_net.yaml')
    hyper_params['epochs'] = 5
    launch_job('hparams', 'test_net.py', hyper_params)