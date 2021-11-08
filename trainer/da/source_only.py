# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

from trainer.base_trainer import BaseTrainer


__all__ = ['SourceOnly']


class SourceOnly(BaseTrainer):
    def __init__(self, cfg):
        super(SourceOnly, self).__init__(cfg)

