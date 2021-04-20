"""Save images into npy format.
"""
import glob
import os
import sys

import imageio
import numpy as np
from tqdm import tqdm

# TODO: argparse instead of argv!!!

if __name__ == "__main__":
    print("Move images in", sys.argv[1], "to", sys.argv[2])

    files = glob.glob(os.path.join(sys.argv[1], "*.png"))
    shape = imageio.imread(files[0]).shape
    data = np.zeros(shape=(len(files), *shape), dtype=np.uint8)

    pbar = tqdm(files)
    pbar.set_description("Move images to npy.")
    for idx, f in enumerate(pbar):
        data[idx] = imageio.imread(f)

    np.save(sys.argv[2], data)
