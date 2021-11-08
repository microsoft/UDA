# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_funcs import entropy_func
from trainer.da.back.dannp import DANNP
import utils.loss as loss
from models import *
from utils.utils import get_coeff


__all__ = ['ToAlign']


class ToAlign(DANNP):
    def __init__(self, cfg):
        super(ToAlign, self).__init__(cfg)

    def build_base_models(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'pretrained': self.cfg.MODEL.PRETRAIN,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
            'hda': True,
            'toalign': True,
        }

        basenet = eval(basenet_name)(**kwargs).cuda()

        return basenet

    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src['image'].cuda(), data_src['label'].cuda()
        inputs_tar, labels_tar = data_tar['image'].cuda(), data_tar['label'].cuda()

        # --------- classification --------------
        outputs_all_src = self.base_net(inputs_src)  # [f, y, z]
        assert len(outputs_all_src) == 3, \
            f'Expected return with size 3, but got {len(outputs_all_src)}'
        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)
        focals_src = outputs_all_src[-1]

        # --------- alignment --------------
        outputs_all_src = self.base_net(inputs_src, toalign=True, labels=labels_src)  # [f_p, y_p, z_p]
        outputs_all_tar = self.base_net(inputs_tar)  # [f, y, z]
        assert len(outputs_all_src) == 3 and len(outputs_all_tar) == 3, \
            f'Expected return with size 3, but got {len(outputs_all_src)}'
        focals_tar = outputs_all_tar[-1]

        logits_all = torch.cat((outputs_all_src[1], outputs_all_tar[1]), dim=0)
        softmax_all = nn.Softmax(dim=1)(logits_all)
        focals_all = torch.cat((focals_src, focals_tar), dim=0)

        ent_tar = entropy_func(nn.Softmax(dim=1)(outputs_all_tar[1].data)).mean()

        # classificaiton loss
        loss_cls_tar = F.cross_entropy(outputs_all_tar[1].data, labels_tar)

        # domain alignment
        loss_alg = loss.d_align_uda(
            softmax_output=softmax_all, d_net=self.d_net,
            coeff=get_coeff(self.ite, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
        )

        # hda
        loss_hda = focals_all.abs().mean()

        loss_ttl = loss_cls_src + loss_alg * self.cfg.METHOD.W_ALG + loss_hda

        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_hda: {loss_hda.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'ent_tar: {ent_tar.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
            # tensorboard
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_cls_tar': loss_cls_tar.item(),
                'l_alg': loss_alg.item(),
                'l_hda': loss_hda.item(),
                'l_ttl': loss_ttl.item(),
                'ent_tar': ent_tar.item(),
            })
