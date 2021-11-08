# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import torch
from torch import Tensor
import torch.nn as nn

from utils.torch_funcs import grl_hook, entropy_func


class WeightBCE(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super(WeightBCE, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor, label: Tensor, weight: Tensor) -> Tensor:
        """
        :param x: [N, 1]
        :param label: [N, 1]
        :param weight: [N, 1]
        """
        label = label.float()
        cross_entropy = - label * torch.log(x + self.epsilon) - (1 - label) * torch.log(1 - x + self.epsilon)
        return torch.sum(cross_entropy * weight.float()) / 2.


def d_align_uda(softmax_output: Tensor, features: Tensor = None, d_net=None,
                coeff: float = None, ent: bool = False):
    loss_func = WeightBCE()

    d_input = softmax_output if features is None else features
    d_output = d_net(d_input, coeff=coeff)
    d_output = torch.sigmoid(d_output)

    batch_size = softmax_output.size(0) // 2
    labels = torch.tensor([[1]] * batch_size + [[0]] * batch_size).long().cuda()  # 2N x 1

    if ent:
        x = softmax_output
        entropy = entropy_func(x)
        entropy.register_hook(grl_hook(coeff))
        entropy = torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[batch_size:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[:batch_size] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

    else:
        weight = torch.ones_like(labels).float() / batch_size

    loss_alg = loss_func.forward(d_output, labels, weight.view(-1, 1))

    return loss_alg


class MMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def _guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # number of used samples
        batch_size = int(source.size()[0])
        kernels = self._guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                        fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss
