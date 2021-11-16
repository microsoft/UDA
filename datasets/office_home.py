# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os

from datasets.common_dataset import CommonDataset
from datasets.reader import read_images_labels


class OfficeHome(CommonDataset):
    """
    -data_root:
     |
     |-art
     |-clipart
     |-product
     |-real_world
       |-Alarm_Clock
         |-0001.jpg
    """
    def __init__(self, data_root, domains: list, status: str = 'train', trim: int = 0):
        super().__init__(is_train=(status == 'train'))

        self._domains = ['product', 'art', 'clipart', 'real_world']

        if domains[0] not in self._domains:
            raise ValueError(f'Expected \'domain\' in {self._domains}, but got {domains[0]}')
        _status = ['train', 'val', 'test']
        if status not in _status:
            raise ValueError(f'Expected \'status\' in {_status}, but got {status}')

        self.image_root = data_root

        # read txt files
        data = read_images_labels(
            os.path.join(f'dataset_map/office_home', f'{domains[0]}.txt'),
            shuffle=(status == 'train'),
            trim=0
        )

        self.data = data
        self.domain_id = [0] * len(self.data)

    def __len__(self):
        return len(self.data)
