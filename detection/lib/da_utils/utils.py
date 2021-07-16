# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import os
import time
import logging


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


def get_parameters(model):
    ps = []
    for p in model.parameters():
        if p.requires_grad:
            ps.append(p)
    return ps
