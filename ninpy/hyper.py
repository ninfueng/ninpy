import time
import glob
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
from typing import Callable

from .config import dict2str, load_yaml
from .job import launch_job


def suggest_hparams(input):
    """Receive an extractable input and extract those information
    as the hyper parameters. For instance expected `trial` for optuna.
    """

    return


def temp_objective(
    input: dict,
    py_script: str,
    basic_yaml: str,
    target_name: str,
    suggest_hparams: Callable,
    is_minimize: bool = False) -> float:
    """Template objective function.
    """
    assert isinstance(py_script, str)
    assert isinstance(basic_yaml, str)
    assert isinstance(target_name, str)
    assert isinstance(is_minimize, bool)

    new_hparams = suggest_hparams(input)
    hypers = load_yaml(basic_yaml)
    for s in suggest_hparams.keys():
        hypers[s] = new_hparams[s]

    exp_pth = dict2str(hypers)
    datetime = time.strftime("%Y:%m:%d-%H:%M:%S")
    exp_pth = f'{datetime}-{exp_pth}'
    if len(exp_pth) > 255:
        exp_pth = exp_pth[0: 254]

    launch_job('job', py_script, hypers, exp_pth)
    result_pth = glob.glob(f'{exp_pth}/results.yaml')
    assert len(result_pth) == 1, f'{result_pth} is more than or less than one.'

    results = load_yaml(result_pth[0])
    target = -float(results['results'][target_name]) if is_minimize else float(results['results'][target_name])
    return target


if __name__ == '__main__':
    NUM_TUNING = 80
    SEED = 2021

    np.random.seed(SEED)
    rstate = np.random.RandomState(SEED)
    trials = Trials()
    choice = ('f', 'b', 't')

    space = {}
    for i in range(18):
        space.update(
            {f'l{i}': hp.choice(f'l{i}', choice)})
    for i in range(3):
        space.update(
            {f's{i}': hp.choice(f's{i}', choice)})

    best = fmin(
        temp_objective,
        space,
        algo=tpe.suggest,
        max_evals=NUM_TUNING,
        trials=trials,
        rstate=rstate)

    df = pd.DataFrame(trials.results)
    df_name = 'asb_runner.csv'
    df.to_csv(df_name, index=None)
    print(df)
    print(f'save this dataframe@{df_name}')

if __name__ == '__main__':
    pass