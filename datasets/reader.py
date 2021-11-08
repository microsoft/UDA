# --------------------------------------------------------
# this code is borrowed from https://github.com/cuishuhao/GVB/blob/master/GVB-GD/data_list.py
# --------------------------------------------------------

import random
import numpy as np


def read_images_labels(image_path, shuffle: bool = False, trim: int = 0) -> list:
    """
    trim: if trim > 0, trim the dataset with size trim (used for domainnet validation)
    return: list[(path_to_image, label), (), ()]
    """
    with open(image_path, 'r') as f:
        data = f.readlines()
    if shuffle:
        random.shuffle(data)
    images_labels = [(val.split()[0], int(val.split()[1])) for val in data]
    if 0 < trim < len(images_labels):
        ids = np.linspace(0, len(images_labels), trim, endpoint=False).astype(int)
        images_labels = [images_labels[i] for i in ids]
    return images_labels
