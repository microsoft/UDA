# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from utils import calc_coeff
from torch_ops import grl_hook


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class Conv2dML(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2dML, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2dML, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2dML, self).forward(x)
        return out


class BatchNorm2dML(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2dML, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var,
                               weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device),
                               weight, bias, training=True, momentum=1)
        return out


class BatchNorm1dML(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1dML, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


class LinearML(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearML, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(LinearML, self).forward(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv2dML(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2dML(planes)
        self.conv2 = Conv2dML(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = BatchNorm2dML(planes)
        self.conv3 = Conv2dML(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2dML(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, gvbg=True, init=False, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # backbone network
        self.conv1 = Conv2dML(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2dML(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.feature_layers = [self.conv1, self.bn1, self.relu, self.maxpool,
                               self.layer1, self.layer2, self.layer3, self.layer4,
                               self.global_avgpool]

        self.fdim = 512 * block.expansion
        # cls
        self.fc = LinearML(self.fdim, num_classes)
        # gvbg
        self.gvbg = gvbg
        if self.gvbg:
            self.gvbg_layer = LinearML(self.fdim, num_classes)

        if init:
            self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dML(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                BatchNorm2dML(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
                nn.init.zeros_(m.bias)

    def output_num(self):
        return self.fdim

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def get_parameters(self):
        feature_layers_params = []
        for m in self.feature_layers:
            feature_layers_params += list(m.parameters())
        if self.gvbg:
            parameter_list = [
                {"params": feature_layers_params, "lr_mult": 1, 'decay_mult': 2},
                {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2},
                {"params": self.gvbg_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [
                {"params": feature_layers_params, "lr_mult": 1, 'decay_mult': 2},
                {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list

    def forward(self, x, return_feature=False, gvbg=False):
        f = self.featuremaps(x)
        f = self.global_avgpool(f)
        f = f.view(f.size(0), -1)

        y = self.fc.forward(f)

        if gvbg:
            z = self.gvbg_layer.forward(f)
        else:
            z = y
        if return_feature:
            return f, y, z
        else:
            return y, z


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, gvbd=False, init=False):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = LinearML(in_feature, hidden_size)
        self.ad_layer2 = LinearML(hidden_size, hidden_size)
        self.ad_layer3 = LinearML(hidden_size, 1)
        if gvbd:
            self.gvbd = LinearML(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.iter_num = 0

        if init:
            self._init_params()

    def forward(self, x, gvbd=True):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1.forward(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)

        if gvbd:
            z = self.gvbd(x)
        else:
            z = y
        return y, z

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

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def init_pretrained_weights(model, pretrain_dict):
    if 'state' in pretrain_dict:
        dict_init = pretrain_dict['state']
    elif 'state_dict' in pretrain_dict:
        dict_init = pretrain_dict['state_dict']
    else:
        dict_init = pretrain_dict
    model_dict = model.state_dict()
    dict_init = {k: v for k, v in dict_init.items() if
                 k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(dict_init)
    model.load_state_dict(model_dict)
    logging.info("    model dict | init keys: {}/{}".format(len(model_dict.keys()), len(dict_init.keys())))


def resnet_advnet(pretrained=True, num_classes=1000, in_dim=1000, gvb=False):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, gvbg=gvb, init=True)
    advnet = AdversarialNetwork(in_dim, 1024, gvbd=gvb, init=True)
    if pretrained:
        init_pretrained_weights(model, model_zoo.load_url(model_urls['resnet50']))
    return model, advnet
