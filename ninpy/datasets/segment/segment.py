from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np


def cvt2mask(mask: np.ndarray, class2colormap: Dict[Any, List[int]]) -> np.ndarray:
    """Convert a PNG mask (3-dimensional) to one-hot masks."""
    # TODO: working to cover both Camvid and Voc2012.
    height, width = mask.shape[:2]
    accum_mask = np.zeros((height, width, len(class2colormap)), dtype=np.float32)
    for label_index, label in enumerate(class2colormap):
        accum_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(np.float32)
    return accum_mask

