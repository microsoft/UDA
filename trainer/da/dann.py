# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_funcs import entropy_func
from trainer.base_trainer import BaseTrainer
from models.discriminator import *
from utils.loss import d_align_uda
from utils.utils import get_coeff


class DANN(BaseTrainer):
    def __init__(self, cfg):
        super(DANN, self).__init__(cfg)

    def build_models(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_models()
        fdim = self.base_net.fdim
        # discriminator
        self.d_net = eval(self.cfg.MODEL.DNET)(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUTDIM
        ).cuda()

        self.registed_models = {'base_net': self.base_net, 'd_net': self.d_net}
        self.model_parameters()
        parameter_list = self.base_net.get_parameters() + self.d_net.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src['image'].cuda(), data_src['label'].cuda()
        inputs_tar, labels_tar = data_tar['image'].cuda(), data_tar['label'].cuda()

        outputs_all_src = self.base_net(inputs_src)
        outputs_all_tar = self.base_net(inputs_tar)

        features_all = torch.cat((outputs_all_src[0], outputs_all_tar[0]), dim=0)
        logits_all = torch.cat((outputs_all_src[1], outputs_all_tar[1]), dim=0)
        softmax_all = nn.Softmax(dim=1)(logits_all)

        ent_tar = entropy_func(nn.Softmax(dim=1)(outputs_all_tar[1].data)).mean()

        # classificaiton
        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)
        loss_cls_tar = F.cross_entropy(outputs_all_tar[1].data, labels_tar)

        # domain alignment
        loss_alg = d_align_uda(
            softmax_all, features_all, self.d_net,
            coeff=get_coeff(self.ite, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
        )

        loss_ttl = loss_cls_src + loss_alg * self.cfg.METHOD.W_ALG

        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'ent_tar: {ent_tar.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
            # tensorboard
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_cls_tar': loss_cls_tar.item(),
                'l_alg': loss_alg.item(),
                'l_ttl': loss_ttl.item(),
                'ent_tar': ent_tar.item(),
            })
