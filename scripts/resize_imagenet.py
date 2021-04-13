"""Script for resize ImageNet type of directories.

Example:
>>> python resize_imagenet.py --path ./imagenet
"""
import argparse
import os
from glob import glob

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Resize ImageNet script.")
parser.add_argument("--path", default="~/datasets/imagenet", type=str)
parser.add_argument("--resize", default=256, type=int)
args = parser.parse_args()


def resize_save_img(path: str, resized_path: str, resize: int) -> None:
    """Resize and save an images.

    Args:
        path (str): ImageNet locations with sub directories: train and val.
        resized_path (str): Save resized ImageNet locations.
        resize (int): Size of wanted image size.
    """
    assert isinstance(path, str)
    assert isinstance(resized_path, str)
    assert isinstance(resize, int)

    basename = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, f"Cannot find an image {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize))
    cv2.imwrite(os.path.join(resized_path, basename), img)


if __name__ == "__main__":
    path = os.path.expanduser(args.path)
    # In case of `imagenet/`, get imagenet from this.
    if path[-1] == "/":
        path = path[:-1]
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    # Resized folder name = imagenet256.
    resized_path = os.path.join(dirname, basename + str(args.resize))
    datasets = ["train", "val"]

    for dataset in datasets:
        class_folders = glob(os.path.join(path, f"{dataset}/n*"))

        for f in tqdm(class_folders):
            class_folder = os.path.basename(f)
            img_paths = glob(os.path.join(f, "*.JPEG"))
            for i in img_paths:
                resized_class_folder = os.path.join(resized_path, dataset, class_folder)
                os.makedirs(resized_class_folder, exist_ok=True)
                resize_save_img(i, resized_class_folder, args.resize)
