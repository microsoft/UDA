# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import time
import json
import random
import logging
import numpy as np
from termcolor import colored

import torch


def write_log(log_path, log):
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')


def create_logger(log_dir):
    # create logger
    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(path, data_dict, ite, snap=False, is_best=False):
    ckpt = data_dict
    if snap:
        save_path = os.path.join(path, f'models-{ite:06d}.pt')
    else:
        save_path = os.path.join(path, f'models-last.pt')
    torch.save(ckpt, save_path)
    logging.info(f'    models saved to {save_path}')
    logging.info(f'    keys: {data_dict.keys()}')
    if is_best:
        save_path = os.path.join(path, f'models-best.pt')
        torch.save(ckpt, save_path)
        logging.info(f'    best models saved to {save_path}')


def init_from_pretrained_weights(model, pretrain_dict, pretrain_exld=[]):
    if 'state' in pretrain_dict:
        dict_init = pretrain_dict['state']
    elif 'state_dict' in pretrain_dict:
        dict_init = pretrain_dict['state_dict']
    elif 'models' in pretrain_dict:
        dict_init = pretrain_dict['models']
    elif 'model' in pretrain_dict:
        dict_init = pretrain_dict['model']
    else:
        dict_init = pretrain_dict
    model_dict = model.state_dict()
    dict_init = {k: v for k, v in dict_init.items() if
                 k in model_dict and v.size() == model_dict[k].size() and k not in pretrain_exld}
    model_dict.update(dict_init)
    model.load_state_dict(model_dict)
    logging.info(f'    models keys / loaded keys: {len(model_dict.keys())}/{len(dict_init.keys())}')


def get_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
