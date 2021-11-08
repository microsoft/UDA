# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

from typing import Any, Type, Union, List, Optional

import torch
import torch.nn as nn

from models.base_model import BaseModel
from utils.utils import init_from_pretrained_weights

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(BaseModel):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 **kwargs
                 ):
        super().__init__(num_classes=num_classes, **kwargs)

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        self._fdim = 512 * block.expansion

        self._init_params()

        # head
        self.build_head()

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_backbone_parameters(self):
        feature_layers_params = []
        for m in self.feature_layers:
            feature_layers_params += list(m.parameters())
        parameter_list = [{'params': feature_layers_params, 'lr_mult': 1}]

        return parameter_list

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_backbone(self, x):
        x = self._forward_impl(x)
        x = self.global_avgpool(x)
        f = torch.flatten(x, 1)
        return f


def resnet18(pretrained: bool = True, num_classes: int = 1000, **kwargs: Any) -> ResNet:
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)
    if pretrained:
        init_from_pretrained_weights(
            model,
            torch.hub.load_state_dict_from_url(url=model_urls['resnet18'], map_location='cpu', model_dir='downloads')
        )

    return model


def resnet34(pretrained: bool = True, num_classes: int = 1000, **kwargs: Any) -> ResNet:
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        init_from_pretrained_weights(
            model,
            torch.hub.load_state_dict_from_url(url=model_urls['resnet34'], map_location='cpu', model_dir='downloads')
        )

    return model


def resnet50(pretrained: bool = True, num_classes: int = 1000, **kwargs: Any) -> ResNet:
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        init_from_pretrained_weights(
            model,
            torch.hub.load_state_dict_from_url(url=model_urls['resnet50'], map_location='cpu', model_dir='downloads')
        )

    return model


def resnet101(pretrained: bool = True, num_classes: int = 1000, **kwargs: Any) -> ResNet:
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if pretrained:
        init_from_pretrained_weights(
            model,
            torch.hub.load_state_dict_from_url(url=model_urls['resnet101'], map_location='cpu', model_dir='downloads')
        )

    return model
