# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import torch.nn as nn

from utils.torch_funcs import init_weights_fc, init_weights_fc0, init_weights_fc1, init_weights_fc2

__all__ = ['BaseModel']


class BaseModel(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 hda: bool = False,  # whether use hda head
                 toalign: bool = False,  # whether use toalign
                 **kwargs
                 ):
        super().__init__()
        self.num_classes = num_classes
        self._fdim = None

        # HDA
        self.hda = hda

        # toalign
        self.toalign = toalign

    def build_head(self):
        # classification head
        self.fc = nn.Linear(self.fdim, self.num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        # HDA head
        if self.hda:
            self.fc.apply(init_weights_fc)
            self.fc0 = nn.Linear(self._fdim, self.num_classes)
            self.fc0.apply(init_weights_fc0)
            self.fc1 = nn.Linear(self._fdim, self.num_classes)
            self.fc1.apply(init_weights_fc1)
            self.fc2 = nn.Linear(self._fdim, self.num_classes)
            self.fc2.apply(init_weights_fc2)

    @property
    def fdim(self) -> int:
        return self._fdim

    def get_backbone_parameters(self):
        return []

    def get_parameters(self):
        parameter_list = self.get_backbone_parameters()
        parameter_list.append({'params': self.fc.parameters(), 'lr_mult': 10})
        if self.hda:
            parameter_list.append({'params': self.fc0.parameters(), 'lr_mult': 10})
            parameter_list.append({'params': self.fc1.parameters(), 'lr_mult': 10})
            parameter_list.append({'params': self.fc2.parameters(), 'lr_mult': 10})

        return parameter_list

    def forward_backbone(self, x):
        """ input x --> output feature """
        return x

    def _get_toalign_weight(self, f, labels=None):
        assert labels is not None, f'labels should be asigned'
        w = self.fc.weight[labels].detach()  # [B, C]
        if self.hda:
            w0 = self.fc0.weight[labels].detach()
            w1 = self.fc1.weight[labels].detach()
            w2 = self.fc2.weight[labels].detach()
            w = w - (w0 + w1 + w2)
        eng_org = (f**2).sum(dim=1, keepdim=True)  # [B, 1]
        eng_aft = ((f*w)**2).sum(dim=1, keepdim=True)  # [B, 1]
        scalar = (eng_org / eng_aft).sqrt()
        w_pos = w * scalar

        return w_pos

    def forward(self, x, toalign=False, labels=None) -> tuple:
        """
        return: [f, y, ...]
        """
        f = self.forward_backbone(x)  # output feature [B, C]
        assert f.dim() == 2, f'Expected dim of returned features to be 2, but found {f.dim()}'

        if toalign:
            w_pos = self._get_toalign_weight(f, labels=labels)
            f_pos = f * w_pos
            y_pos = self.fc(f_pos)
            if self.hda:
                z_pos0 = self.fc0(f_pos)
                z_pos1 = self.fc1(f_pos)
                z_pos2 = self.fc2(f_pos)
                z_pos = z_pos0 + z_pos1 + z_pos2
                return f_pos, y_pos - z_pos, z_pos
            else:
                return f_pos, y_pos
        else:
            y = self.fc(f)
            if self.hda:
                z0 = self.fc0(f)
                z1 = self.fc1(f)
                z2 = self.fc2(f)
                z = z0 + z1 + z2
                return f, y - z, z
            else:
                return f, y
