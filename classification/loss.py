# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np
import torch

from torch_ops import grl_hook, entropy_func, WeightBCE


def DA(softmax_output, ad_net, coeff=None, features=None, non_weight=False, gvb=False):
    mean_entropy = 0.
    weightbce = WeightBCE()
    if features is None:
        ad_out, fc_out = ad_net(softmax_output, gvbd=gvb)
    else:
        ad_out, fc_out = ad_net(features, gvbd=gvb)
    ad_out = ad_out - fc_out if gvb else ad_out
    ad_out = torch.sigmoid(ad_out)
    gvbd = torch.mean(torch.abs(fc_out))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()  # 2N x 1

    if non_weight:
        weight = torch.ones_like(dc_target)
    else:
        x = softmax_output
        entropy = entropy_func(x)
        entropy.register_hook(grl_hook(coeff))
        entropy = torch.exp(-entropy)
        mean_entropy = torch.mean(entropy)

        # for each smaple, obtain weight based on the entropy
        source_mask = torch.ones_like(entropy)
        source_mask[batch_size:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:batch_size] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
    loss_trs = weightbce.forward(ad_out, dc_target, weight.view(-1, 1))
    return loss_trs, mean_entropy, gvbd
