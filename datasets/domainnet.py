# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import random
import functools
import operator

from datasets.common_dataset import CommonDataset
from datasets.reader import read_images_labels


class DomainNet(CommonDataset):
    """
    -data_root:
     |
     |-clipart
     |-infograph
     |-painting
     |-quickdraw
     |-real
     |-sketch
       |-dog
         |-*.jpg
    """
    def __init__(self, data_root, domains: list, status: str = 'train', trim: int = 0):
        super().__init__(is_train=(status == 'train'))

        self._domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

        if not set(domains).issubset(self._domains):
            raise ValueError(f'Expected \'domains\' in {self._domains}, but got {domains}')
        _status = ['train', 'val', 'test']
        if status not in _status:
            raise ValueError(f'Expected \'status\' in {_status}, but got {status}')

        self.image_root = data_root

        self.data = []

        # read txt files
        data = []
        len_domains = []
        for _domain in domains:
            suffix = 'train' if status == 'train' else 'test'
            _data = read_images_labels(
                os.path.join(f'dataset_map/domainnet', f'{domains[0]}_{suffix}.txt'),
                shuffle=(status == 'train'),
                trim=trim
            )
            len_domains.append(len(_data))
            data.append(_data)

        max_len = max(len_domains)
        # keep all domains have same # data | training
        if status == 'train':
            for i in range(len(data)):
                data[i] = data[i] + random.sample(data[i], max_len - len(data[i]))
        # cat to one domain
        self.data = functools.reduce(operator.iconcat, data, [])

        self.num_domain = len(domains)
        domain_id = [[i] * len_domains[i] for i in range(self.num_domain)]
        self.domain_id = functools.reduce(operator.iconcat, domain_id, [])

        assert len(self.domain_id) == len(self.data), f'domain_id not match data'

    def __len__(self):
        return len(self.data)
