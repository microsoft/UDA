# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import torch.nn as nn

from utils.torch_funcs import grl_hook

__all__ = ['Discriminator']


class Discriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, out_feature: int = 1):
        super(Discriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, out_feature)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self._init_params()

    def forward(self, x, coeff: float):
        x = x * 1.  # to avoid affect the grad from another pipeline to x_0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1.forward(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)

        return y

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1., 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 10}]
