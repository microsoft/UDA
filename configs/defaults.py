# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os

from yacs.config import CfgNode as CN


_C = CN()
_C.SEED = 123
_C.WORKERS = 8
_C.TRAINER = 'Trainer'


# tasks
_C.TASK = CN()
_C.TASK.NAME = 'UDA'
_C.TASK.SSDA_SHOT = 1

# ================= training ====================
_C.TRAIN = CN()
_C.TRAIN.TEST_FREQ = 500
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.SAVE_FREQ = 5000
_C.TRAIN.TTL_ITE = 8000

_C.TRAIN.BATCH_SIZE_SOURCE = 36
_C.TRAIN.BATCH_SIZE_TARGET = 36
_C.TRAIN.BATCH_SIZE_TEST = 36
_C.TRAIN.LR = 0.001

_C.TRAIN.OUTPUT_ROOT = 'temp'
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.OUTPUT_LOG = 'log'
_C.TRAIN.OUTPUT_TB = 'tensorboard'
_C.TRAIN.OUTPUT_CKPT = 'ckpt'
_C.TRAIN.OUTPUT_RESFILE = 'log.txt'

# ================= models ====================
_C.OPTIM = CN()
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9

# ================= models ====================
_C.MODEL = CN()
_C.MODEL.PRETRAIN = True
_C.MODEL.BASENET = 'resent50'
_C.MODEL.BASENET_DOMAIN_EBD = False  # for domain embedding for transformer
_C.MODEL.DNET = 'Discriminator'
_C.MODEL.D_INDIM = 0
_C.MODEL.D_OUTDIM = 1
_C.MODEL.D_HIDDEN_SIZE = 1024
_C.MODEL.D_WGAN_CLIP = 0.01
_C.MODEL.VIT_DPR = 0.1
_C.MODEL.VIT_USE_CLS_TOKEN = True
_C.MODEL.VIT_PRETRAIN_EXLD = []
# extra layer
_C.MODEL.EXT_LAYER = False
_C.MODEL.EXT_NUM_TOKENS = 100
_C.MODEL.EXT_NUM_LAYERS = 1
_C.MODEL.EXT_NUM_HEADS = 24
_C.MODEL.EXT_LR = 10.
_C.MODEL.EXT_DPR = 0.1
_C.MODEL.EXT_SKIP = True
_C.MODEL.EXT_FEATURE = 768

# ================= dataset ====================
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NAME = 'office_home'
_C.DATASET.SOURCE = []
_C.DATASET.TARGET = []
_C.DATASET.TRIM = 0

# ================= method ====================
_C.METHOD = CN()
_C.METHOD.W_ALG = 1.0
_C.METHOD.ENT = False

# HDA
_C.METHOD.HDA = CN()
_C.METHOD.HDA.W_HDA = 1.0


def get_default_and_update_cfg(args):
    cfg = _C.clone()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    #
    cfg.SEED = args.seed

    #
    if args.data_root:
        cfg.DATASET.ROOT = args.data_root

    # dataset maps
    maps = {
        'office_home': {
            'p': 'product',
            'a': 'art',
            'c': 'clipart',
            'r': 'real_world'
        }
    }

    # source & target
    cfg.DATASET.SOURCE = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in args.source]
    cfg.DATASET.TARGET = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in args.target]

    # class
    if cfg.DATASET.NAME == 'office_home':
        cfg.DATASET.NUM_CLASSES = 65
    elif cfg.DATASET.NAME == 'office':
        cfg.DATASET.NUM_CLASSES = 31
    elif cfg.DATASET.NAME == 'visda-2017':
        cfg.DATASET.NUM_CLASSES = 12
    elif cfg.DATASET.NAME == 'domainnet' or cfg.DATASET.NAME == 'uda_domainnet':
        cfg.DATASET.NUM_CLASSES = 345
    elif cfg.DATASET.NAME == 'ssda-domainnet':
        cfg.DATASET.NUM_CLASSES = 126
    else:
        raise NotImplementedError(f'cfg.DATASET.NAME: {cfg.DATASET.NAME} not imeplemented')

    # output
    if args.output_root:
        cfg.TRAIN.OUTPUT_ROOT = args.output_root
    if args.output_dir:
        cfg.TRAIN.OUTPUT_DIR = args.output_dir
    else:
        cfg.TRAIN.OUTPUT_DIR = '_'.join(cfg.DATASET.SOURCE) + '2' + '_'.join(cfg.DATASET.TARGET) + '_' + str(args.seed)

    #
    cfg.TRAIN.OUTPUT_CKPT = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'ckpt', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_LOG = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'log', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_TB = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'tensorboard', cfg.TRAIN.OUTPUT_DIR)
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_TB, exist_ok=True)
    cfg.TRAIN.OUTPUT_RESFILE = os.path.join(cfg.TRAIN.OUTPUT_LOG, 'log.txt')

    return cfg


def check_cfg(cfg):
    # OUTPUT
    cfg.TRAIN.OUTPUT_CKPT = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'ckpt', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_LOG = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'log', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_TB = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'tensorboard', cfg.TRAIN.OUTPUT_DIR)
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_TB, exist_ok=True)
    cfg.TRAIN.OUTPUT_RESFILE = os.path.join(cfg.TRAIN.OUTPUT_LOG, 'log.txt')

    # dataset
    maps = {
        'office_home': {
            'p': 'product',
            'a': 'art',
            'c': 'clipart',
            'r': 'real_world'
        }
    }
    cfg.DATASET.SOURCE = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in cfg.DATASET.SOURCE]
    cfg.DATASET.TARGET = [maps[cfg.DATASET.NAME][_d] if _d in maps[cfg.DATASET.NAME].keys() else _
                          for _d in cfg.DATASET.TARGET]

    datapath_list = {
        'office-home': {
            'p': ['Product.txt', 'Product.txt'],
            'a': ['Art.txt', 'Art.txt'],
            'c': ['Clipart.txt', 'Clipart.txt'],
            'r': ['Real_World.txt', 'Real_World.txt']
        },
        'uda_domainnet': {
            'c': ['clipart_train.txt', 'clipart_test.txt'],
            'i': ['infograph_train.txt', 'infograph_test.txt'],
            'p': ['painting_train.txt', 'painting_test.txt'],
            'q': ['quickdraw_train.txt', 'quickdraw_test.txt'],
            'r': ['real_train.txt', 'real_test.txt'],
            's': ['sketch_train.txt', 'sketch_test.txt']
        },
        'visda-2017': {
            't': 'train_list.txt',
            'v': 'validation_list.txt'
        },
        'office': {
            'a': 'amazon.txt',
            'd': 'dslr.txt',
            'w': 'webcam.txt'
        },
        'domainnet': {
            'c': 'clipart_{}.txt',
            'i': 'infograph_{}.txt',
            'p': 'painting_{}.txt',
            'q': 'quickdraw_{}.txt',
            'r': 'real_{}.txt',
            's': 'sketch_{}.txt',
        },
        'ssda-domainnet': {
            'c': 'clipart',
            'p': 'painting',
            'r': 'real',
            's': 'sketch'
        }
    }

    # class
    if cfg.DATASET.NAME == 'office_home':
        cfg.DATASET.NUM_CLASSES = 65
    elif cfg.DATASET.NAME == 'office':
        cfg.DATASET.NUM_CLASSES = 31
    elif cfg.DATASET.NAME == 'visda-2017':
        cfg.DATASET.NUM_CLASSES = 12
    elif cfg.DATASET.NAME == 'domainnet' or cfg.DATASET.NAME == 'uda_domainnet':
        cfg.DATASET.NUM_CLASSES = 345
    elif cfg.DATASET.NAME == 'ssda-domainnet':
        cfg.DATASET.NUM_CLASSES = 126
    else:
        raise NotImplementedError(f'cfg.DATASET.NAME: {cfg.DATASET.NAME} not imeplemented')

    return cfg
