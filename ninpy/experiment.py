#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import glob
import os
import shutil
import sys
from functools import reduce


def set_experiment(
    exp_pth: str, match_list=["*.py", "*.sh", "*.yaml"], rm_exist: bool = True
) -> str:
    r"""
    Modified:
        https://github.com/VITA-Group/FasterSeg/blob/master/tools/utils/darts_utils.py
    Create a folder with the name f'{datetime}-{experiment_path}' format.
    With scripts folder into it and copy all scripts within `list_types` into this folder.
    WARNING: MUST NOT APPLY THIS FUNCTION WITH LOGGING!.
    Example:
    >>> set_experiment(experiment_path)
    """
    assert isinstance(exp_pth, str)
    assert len(match_list) > 0

    if rm_exist and os.path.exists(exp_pth):
        shutil.rmtree(exp_pth)
    if not os.path.exists(exp_pth):
        if sys.platform == "win32":
            # For OSError: [WinError 123] The filename,
            # directory name, or volume label syntax is incorrect:
            exp_pth = exp_pth.replace(".", "_")
            exp_pth = exp_pth.replace(":", "_")

        os.mkdir(exp_pth)
        # Can have OSError: [Errno 36] File name too long.
        # If so then, bash allows 255 charactors, therefore limit to that amount.
        os.makedirs(os.path.join(exp_pth, "scripts"), exist_ok=True)

    save_scripts = [glob.glob(t, recursive=False) for t in match_list]
    save_scripts = reduce(lambda x, y: x + y, save_scripts)
    for s in save_scripts:
        dst_pth = os.path.join(exp_pth, "scripts", str(s))
        shutil.copyfile(s, dst_pth)
    return exp_pth


if __name__ == "__main__":
    set_experiment("experiment", ["*.py", "*yaml"])
