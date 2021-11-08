# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import json
import argparse

import torch.backends.cudnn as cudnn

from configs.defaults import get_default_and_update_cfg
from utils.utils import create_logger, set_seed
from trainer import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg',      default='configs/test.yaml', type=str)
    parser.add_argument('--seed',   default=123, type=int)
    parser.add_argument('--source', default=None, nargs='+', help='source domain names')
    parser.add_argument('--target', default=None, nargs='+', help='target domain names')
    parser.add_argument('--output_root', default=None, type=str, help='output root path')
    parser.add_argument('--output_dir',  default=None, type=str, help='output path, subdir under output_root')
    parser.add_argument('--data_root',   default=None, type=str, help='path to dataset root')
    parser.add_argument('--opts',   default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg), 'cfg file: {} not found'.format(args.cfg)

    return args


def main():
    args = parse_args()
    cfg = get_default_and_update_cfg(args)
    cfg.freeze()

    # seed
    set_seed(cfg.SEED)

    cudnn.deterministic = True
    cudnn.benchmark = False

    # logger
    logger = create_logger(cfg.TRAIN.OUTPUT_LOG)

    logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))
    logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))

    trainer = eval(cfg.TRAINER)(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
