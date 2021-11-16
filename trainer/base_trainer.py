# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter

from utils.utils import save_model, write_log
from utils.lr_scheduler import inv_lr_scheduler
from datasets import *
from models import *


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

        logging.info(f'--> trainer: {self.__class__.__name__}')

        self.setup()
        self.build_datasets()
        self.build_models()
        self.resume_from_ckpt()

    def setup(self):
        self.start_ite = 0
        self.ite = 0
        self.best_acc = 0.
        self.tb_writer = SummaryWriter(self.cfg.TRAIN.OUTPUT_TB)

    def build_datasets(self):
        logging.info(f'--> building dataset from: {self.cfg.DATASET.NAME}')
        self.dataset_loaders = {}

        # dataset loaders
        if self.cfg.DATASET.NAME == 'office_home':
            dataset = OfficeHome
        elif self.cfg.DATASET.NAME == 'domainnet':
            dataset = DomainNet
        else:
            raise ValueError(f'Dataset {self.cfg.DATASET.NAME} not found')

        self.dataset_loaders['source_train'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='train'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_SOURCE,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['source_test'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.SOURCE, status='val', trim=self.cfg.DATASET.TRIM),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.dataset_loaders['target_train'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='train'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TARGET,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['target_test'] = DataLoader(
            dataset(self.cfg.DATASET.ROOT, self.cfg.DATASET.TARGET, status='test'),
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )
        self.len_src = len(self.dataset_loaders['source_train'])
        self.len_tar = len(self.dataset_loaders['target_train'])
        logging.info(f'    source {self.cfg.DATASET.SOURCE}: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')
        logging.info(f'    target {self.cfg.DATASET.TARGET}: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def build_models(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        self.base_net = self.build_base_models()
        self.registed_models = {'base_net': self.base_net}
        parameter_list = self.base_net.get_parameters()
        self.model_parameters()
        self.build_optim(parameter_list)

    def build_base_models(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'pretrained': self.cfg.MODEL.PRETRAIN,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
        }

        basenet = eval(basenet_name)(**kwargs).cuda()

        return basenet

    def model_parameters(self):
        for k, v in self.registed_models.items():
            logging.info(f'    {k} paras: '
                         f'{(sum(p.numel() for p in v.parameters()) / 1e6):.2f}M')

    def build_optim(self, parameter_list: list):
        self.optimizer = optim.SGD(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            momentum=self.cfg.OPTIM.MOMENTUM,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            nesterov=True
        )
        self.lr_scheduler = inv_lr_scheduler

    def resume_from_ckpt(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-last.pt')
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt)
            for k, v in self.registed_models.items():
                v.load_state_dict(ckpt[k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_ite = ckpt['ite']
            self.best_acc = ckpt['best_acc']
            logging.info(f'> loading ckpt from {last_ckpt} | ite: {self.start_ite} | best_acc: {self.best_acc:.3f}')
        else:
            logging.info('--> training from scratch')

    def train(self):
        # start training
        for _, v in self.registed_models.items():
            v.train()
        for self.ite in range(self.start_ite, self.cfg.TRAIN.TTL_ITE):
            # test
            if self.ite % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.ite != self.start_ite:
                self.base_net.eval()
                self.test()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                ite_rate=self.ite / self.cfg.TRAIN.TTL_ITE * self.cfg.METHOD.HDA.LR_MULT,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
            if self.ite % self.len_src == 0 or self.ite == self.start_ite:
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.ite % self.len_tar == 0 or self.ite == self.start_ite:
                iter_tar = iter(self.dataset_loaders['target_train'])

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
            self.one_step(data_src, data_tar)
            if self.ite % self.cfg.TRAIN.SAVE_FREQ == 0 and self.ite != 0:
                self.save_model(is_best=False, snap=True)

    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src['image'].cuda(), data_src['label'].cuda()

        outputs_all_src = self.base_net(inputs_src)  # [f, y]

        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)

        loss_ttl = loss_cls_src

        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
            # tensorboard
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_ttl': loss_ttl.item(),
            })

    def display(self, data: list):
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | lr: {self.current_lr:.5f} '
        # update
        for _str in data:
            log_str += '| {} '.format(_str)
        logging.info(log_str)

    def update_tb(self, data: dict):
        for k, v in data.items():
            self.tb_writer.add_scalar(k, v, self.ite)

    def step(self, loss_ttl):
        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()

    def test(self):
        logging.info('--> testing on source_test')
        src_acc = self.test_func(self.dataset_loaders['source_test'], self.base_net)
        logging.info('--> testing on target_test')
        tar_acc = self.test_func(self.dataset_loaders['target_test'], self.base_net)
        is_best = False
        if tar_acc > self.best_acc:
            self.best_acc = tar_acc
            is_best = True

        # display
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | src_acc: {src_acc:.3f} | tar_acc: {tar_acc:.3f} | ' \
                  f'best_acc: {self.best_acc:.3f}'
        logging.info(log_str)

        # save results
        log_dict = {
            'I': self.ite,
            'src_acc': src_acc,
            'tar_acc': tar_acc,
            'best_acc': self.best_acc
        }
        write_log(self.cfg.TRAIN.OUTPUT_RESFILE, log_dict)

        # tensorboard
        self.tb_writer.add_scalar('tar_acc', tar_acc, self.ite)
        self.tb_writer.add_scalar('src_acc', src_acc, self.ite)

        self.save_model(is_best=is_best)

    def test_func(self, loader, model):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)
            accs = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), accs.avg))
                data = iter_test.__next__()
                inputs, labels = data['image'].cuda(), data['label'].cuda()
                outputs_all = model(inputs)  # [f, y, ...]
                outputs = outputs_all[1]

                acc = accuracy(outputs, labels)[0]
                accs.update(acc.item(), labels.size(0))

        return accs.avg

    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'ite': self.ite,
            'best_acc': self.best_acc
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, ite=self.ite, is_best=is_best, snap=snap)
