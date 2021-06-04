from collections import OrderedDict
from typing import Any, Dict, List, OrderedDict

import numpy as np
import cv2

# Pipeline cvt colormap -> save with processed classes to location files (not need to process second time).


def cvt_colormap2mask(
    colormap: np.ndarray, class_colormap: OrderedDict[Any, List[int]]
) -> np.ndarray:
    """Convert a PNG colormap (3-dimensional) to one-hot masks.
    Expect to OrderedDict type of input."""
    # TODO: make this generalizes across different datasets.
    height, width = colormap.shape[:2]
    accum_mask = np.zeros((height, width, len(class_colormap)), dtype=np.float32)
    for label_index, label in enumerate(class_colormap):
        accum_mask[:, :, label_index] = np.all(colormap == label, axis=-1).astype(
            np.float32
        )
    return accum_mask


def save_mask(save_dir: str, masks: List[Any]) -> None:
    # TODO:
    assert isinstance(save_dir, str)
    for m in masks:
        pass
    # cv2.imwrite()
    return
