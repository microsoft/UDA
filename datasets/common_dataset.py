# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import random

from PIL import Image
from PIL import ImageFile

from torch.utils.data import Dataset
from .transforms import transform_train, transform_test


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CommonDataset(Dataset):
    def __init__(self, is_train: bool = True):
        self.data = []
        self.domain_id = []
        self.image_root = ''
        self.transform = transform_train() if is_train else transform_test()
        self._domains = None
        self.num_domain = 1

    @property
    def domains(self):
        return self._domains

    def __getitem__(self, index):
        # domain = random.randint(0, self.num_domain - 1)
        # path, label = self.data[domain][index]
        domain = self.domain_id[index]
        path, label = self.data[index]
        path = os.path.join(self.image_root, path)
        with Image.open(path) as image:
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'domain': domain
        }

    def __len__(self):
        pass
