#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ninnart Fuengfusin
"""
import os
import glob
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class ImageResizer(object):
    """Given the directory of image dataset.
    """
    def __init__(self, filetype: str = '*.png'):
        assert isinstance(filetype, str)
        self.list_imgs = glob.glob(filetype)
        list_name_imgs = [Path(i).stem + Path(i).suffix for i in self.list_imgs]
        dir_loc = Path(filetype).parent
        self.save_loc = str(dir_loc) + '_resize'
        self.list_save_loc = [os.path.join(self.save_loc, str(img_loc)) for img_loc in list_name_imgs]

    def gen_resize_imgs(self, resize=(), interpolation=cv2.INTER_LINEAR) -> None:
        assert len(resize) == 2
        if not os.path.exists(self.save_loc):
            os.mkdir(self.save_loc)
        for load, save in zip(self.list_imgs, self.list_save_loc):
            img = cv2.imread(load, cv2.IMREAD_COLOR)
            assert img is not None
            resized_img = cv2.resize(img, resize, interpolation=interpolation)
            cv2.imwrite(save, resized_img)

    def show_img(self, idx: int = None) -> None:
        if idx is None:
            idx = np.random.randint(len(self.list_imgs))
        else:
            assert 0 <= idx < len(self.list_imgs)
        example_path = self.list_imgs[idx]
        example_img = cv2.imread(example_path, cv2.IMREAD_COLOR)
        plt.imshow(example_img)
        plt.title(example_path)
        plt.show()


if __name__ == '__main__':
    img_resize = ImageResizer('./imgs/*.png')
    img_resize.show_img()
    img_resize.gen_resize_imgs((256, 256))