# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------


import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_cross_entropy(input, target, mask):
    log_probility = F.log_softmax(input, dim=1)
    cross_entropy = -log_probility.index_select(-1, target).diag()
    mask_cross_entropy = mask.unsqueeze(1) * cross_entropy

    return mask_cross_entropy.mean()


class WeightBCE(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(WeightBCE, self).__init__()
        self.epsilon = epsilon
        return

    def forward(self, input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) - (1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def entropy_func(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def get_parameters(model1, model2, name=False):
    params = []
    if name:
        for n, p in model1.named_parameters():
            params.append([n, p])
        for n, p in model2.named_parameters():
            params.append([n, p])
    else:
        for n, p in model1.named_parameters():
            params.append(p)
        for n, p in model2.named_parameters():
            params.append(p)

    return params


def generate_pseudo_labels(model, data_loader, num_data, thres=0.9):
    model.train(False)
    with torch.no_grad():
        iter_loader = iter(data_loader)
        labels, labels_mask = np.zeros(num_data), np.zeros(num_data)
        for i in range(len(data_loader)):
            data = iter_loader.next()
            inputs, _, idxs = data[0], data[1], data[2]
            outputs = model(inputs.cuda())
            outputs = F.softmax(outputs, dim=1)
            pred = outputs.max(1)[1].data.cpu().numpy()
            mask = (outputs.max(1)[0].data.cpu().numpy() >= thres) * 1.
            labels[idxs] = pred
            labels_mask[idxs] = mask
        logging.info("==> {}/{}".format(labels_mask.sum(), num_data))
    return labels, labels_mask


def image_classification_test(loader, model, gvb=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.__next__()
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs_ = model(inputs, gvbg=gvb)
            if gvb:
                outputs = outputs_[0] - outputs_[1]
            else:
                outputs = outputs_[0]
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy
