# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch import Tensor


def grl_hook(coeff):
    def func_(grad):
        return -coeff * grad.clone()

    return func_


def entropy_func(x: Tensor) -> Tensor:
    """
    x: [N, C]
    return: entropy: [N,]
    """
    epsilon = 1e-5
    entropy = -x * torch.log(x + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def get_relation(x: Tensor) -> Tensor:
    """
    :param x: [B, L, C]
    :return: [B, L, L]
    """
    x1 = x @ x.transpose(1, 2)
    x_norm = x.norm(dim=-1, keepdim=True)  # [B, L, 1]
    x2 = x_norm @ x_norm.transpose(1, 2)  # [B, L, L]
    x2 = torch.max(x2, torch.ones_like(x2) * 1e-8)

    return x1 / x2


# initialization used only for HDA
def init_weights_fc(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=100)
        nn.init.zeros_(m.bias)


def init_weights_fc0(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


def init_weights_fc1(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1)
        nn.init.zeros_(m.bias)


def init_weights_fc2(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=2)
        nn.init.zeros_(m.bias)

