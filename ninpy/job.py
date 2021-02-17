#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import time
import sys
import os
import logging
from subprocess import check_call
from .config import dump_yaml, load_yaml, dump_yaml


def launch_job(
        job_name: str,
        script_name: str,
        hyper_params: dict,
        exp_pth: str,
        verbose: bool = True) -> None:

    r"""Modified from: https://github.com/cs230-stanford/cs230-code-examples
    Execute Python scripts on demand.
    Designed to use with hyper parameter tuner such as hyperopt and optuna.
    Required scripts with with `yaml` and `exp_pth` option.
    Example:
    ```
    >>> launch_job('hparams', 'test_net.py', {'epochs': 1})
    ```
    """
    # Launch training with this config
    assert isinstance(hyper_params, dict)
    assert isinstance(exp_pth, str)

    PYTHON = sys.executable
    # For saving yaml.
    hyper_yaml = f'{job_name}.yaml'
    if os.path.exists(hyper_yaml):
        os.remove(hyper_yaml)
    dump_yaml(hyper_params, hyper_yaml)
    cmd = f'{PYTHON} {script_name} --yaml={hyper_yaml} --exp_pth={exp_pth}'

    if verbose:
        logging.info(f'Using launch job with: {cmd}')
    # Run command at shell.
    check_call(cmd, shell=True)
    # Remove yaml file after job done.
    os.remove(hyper_yaml)


if __name__ == '__main__':
    hyper_params = {'epochs': 1}
    dump_yaml('test.yaml')
    assert hyper_params['epochs'] == 1

    hyper_params = load_yaml('test.yaml')
    hyper_params.epochs = 5
    assert hyper_params.epochs  == 5

    launch_job('hparams', 'test_net.py', hyper_params)
    os.remove('test.yaml')