# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import time
import os
import logging
from pprint import pformat
import numpy as np

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def write_log(log_path, log):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def create_logger(log_dir):
    # set up logger
    if not os.path.isdir(log_dir):
        logging.info('==> creating {}'.format(log_dir))
        os.makedirs(log_dir)

    time_str = time.strftime('%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = log_dir + '/' + log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=final_log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def save_model(path, data_dict, ite, snap=False, is_best=False):
    ckpt = data_dict
    if snap:
        save_path = os.path.join(path, "model.pth.tar-{:05d}".format(ite))
    else:
        save_path = os.path.join(path, "model.pth.tar-last")
    torch.save(ckpt, save_path)
    logging.info("    model saved to {}".format(save_path))
    logging.info("    keys: {}".format(pformat(data_dict.keys())))
    if is_best:
        save_path = os.path.join(path, "model.pth.tar-best")
        torch.save(ckpt, save_path)
        logging.info("    model saved to {}".format(save_path))


def compute_accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
