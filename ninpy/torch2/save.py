from typing import Union
import logging
from pathlib import Path

import numpy as np
import torch


def save_bin(
    name: str, tensor: Union[torch.Tensor, np.ndarray, float, int]
) -> None:
    """Save bin file with a tensor."""
    assert isinstance(name, str)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if isinstance(tensor, float):
        tensor = np.asarray(tensor, dtype=np.float32)
    elif isinstance(tensor, int):
        tensor = np.asarray(tensor, dtype=np.int32)
    tensor.tofile(name, format="float32")


def np2cpp(
    array: np.ndarray,
    type_var: str,
    name_var: str,
    name_file: str,
    mode: str,
    header_guard: bool = True,
    verbose: bool = True,
) -> None:

    r"""Convert Python 1-4 dimensional array into cpp array.
    TODO: Adding comments sections (adding string) at header file.
    Args:
        array: A Python numpy 1-4D array.
        type_var (str): A string with C++ type 'float', 'int', ..
        name_var (str): A string assigned to name of variable in C++ environment.
        name_file (str): A string assigned to name of C++ file.
        mode (str): A string assigned to mode of open(, ): 'w', 'a'.
        header_guard (bool): Generate header guard.
    Raise:
        NotImplementedError: For array more dimension than 4.
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    array = np.array(array)
    assert isinstance(type_var, str)
    assert isinstance(name_var, str)
    assert isinstance(name_file, str)
    assert isinstance(mode, str)
    assert isinstance(header_guard, bool)
    assert mode in ["w", "a"]

    # Get stem and suffix from name_file.
    name_stem = Path(name_file)
    name_stem = name_stem.stem + name_stem.suffix
    with open(name_file, mode) as file:
        if header_guard:
            # Generate guard band for cpp header file.
            header_name = name_stem.upper()
            file.write(f"#ifndef __{header_name}__\n".replace(".", "_"))
            file.write(f"#define __{header_name}__\n".replace(".", "_"))
        file.write("\n")

        if len(array.shape) == 1:
            file.write(type_var + " " + name_var + str([array.shape[0]]) + " {")
            for i in range(array.shape[0]):
                file.write(str(array[i]))
                file.write(",")

        elif len(array.shape) == 2:
            file.write(
                type_var
                + " "
                + name_var
                + str([array.shape[0]])
                + str([array.shape[1]])
                + " {"
            )
            for i in range(array.shape[0]):
                file.write("{")
                for j in range(array.shape[1]):
                    file.write(str(array[i][j]))
                    file.write(",")
                file.write("},")

        elif len(array.shape) == 3:
            file.write(
                type_var
                + " "
                + name_var
                + str([array.shape[0]])
                + str([array.shape[1]])
                + str([array.shape[2]])
                + " {"
            )
            for i in range(array.shape[0]):
                file.write("{")
                for j in range(array.shape[1]):
                    file.write("{")
                    for k in range(array.shape[2]):
                        file.write(str(array[i][j][k]))
                        file.write(",")
                    file.write("},")
                file.write("},")

        elif len(array.shape) == 4:
            file.write(
                type_var
                + " "
                + name_var
                + str([array.shape[0]])
                + str([array.shape[1]])
                + str([array.shape[2]])
                + str([array.shape[3]])
                + " {"
            )
            for i in range(array.shape[0]):
                file.write("{")
                for j in range(array.shape[1]):
                    file.write("{")
                    for k in range(array.shape[2]):
                        file.write("{")
                        for l in range(array.shape[3]):
                            file.write(str(array[i][j][k][l]))
                            file.write(",")
                        file.write("},")
                    file.write("},")
                file.write("},")

        else:
            raise NotImplementedError(
                "array can have dimensional from 1-4."
                f"However you input has shape as {len(array.shape)}"
            )

        file.write("};\n")
        if header_guard:
            # Generate guard band for cpp header file.
            file.write("\n")
            file.write("#endif")

        if verbose:
            logging.info(
                f"Generate header file: {name_file}"
                f"with guard band {header_guard}."
            )
