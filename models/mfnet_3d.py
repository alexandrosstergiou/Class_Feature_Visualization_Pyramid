# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:04:54 2018

Original Author: Yunpeng Chen
https://github.com/cypw/PyTorch-MFNet/blob/master/network/mfnet_3d.py

@author: George
"""
import logging
from collections import OrderedDict
import torch
import torch.nn as nn


def xavier(_net):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname in ['Sequential', 'AvgPool3d', 'MaxPool3d', 'Dropout', 'ReLU', 'Softmax', 'BnActConv3d'] \
                or 'Block' in classname:
            pass
        else:
            if classname != classname.upper():
                print("Initializer:: '{}' is uninitialized.".format(classname))
    _net.apply(weights_init)


class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):

    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1,1,1), first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid/4)
        kt,pt = (3,1) if use_3d else (1,0)
        # prepare input
        self.conv_i1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1,1,1), pad=(0,0,0))
        self.conv_i2 =     BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1,1,1), pad=(0,0,0))
        # main part
        self.conv_m1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt,3,3), pad=(pt,1,1), stride=stride, g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,3,3), pad=(0,1,1), g=g)
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)

    def forward(self, x):
        _activations = []
        _weights = []

        if isinstance(x, tuple):
            x, _weights, _activations = x

        h = self.conv_i1(x)
        # _activations.append(h)
        # _weights.append(self.conv_i1.conv.weight.data)

        hi2 = self.conv_i2(h)
        # _activations.append(hi2)
        # _weights.append(self.conv_i2.conv.weight.data)

        x_in = x + hi2

        h = self.conv_m1(x_in)
        # _activations.append(h)
        # _weights.append(self.conv_m1.conv.weight.data)

        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        h = h + x
        _activations.append(h)
        # if self.conv_m2.conv.groups:
        #     kernel = self.convm2.conv.weight.data
        #     torch.
        _weights.append(self.conv_m2.conv.weight.data)

        return h, _weights, _activations


class MFNET_3D(nn.Module):

    def __init__(self, num_classes, dropout=None, pretrained=False, pretrained_model="", **kwargs):
        super(MFNET_3D, self).__init__()

        groups = 16
        k_sec = {2: 3,
                 3: 4,
                 4: 6,
                 5: 3}

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d( 3, conv1_num_out, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2), bias=False)),
                    ('bn', nn.BatchNorm3d(conv1_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # conv2 - x56 (x8)
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv1_num_out if i==1 else conv2_num_out,
                                        num_mid=num_mid,
                                        num_out=conv2_num_out,
                                        stride=(2,1,1) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[2]+1)
                    ]))

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv2_num_out if i==1 else conv3_num_out,
                                        num_mid=num_mid,
                                        num_out=conv3_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[3]+1)
                    ]))

        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv3_num_out if i==1 else conv4_num_out,
                                        num_mid=num_mid,
                                        num_out=conv4_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[4]+1)
                    ]))

        # conv5 - x7 (x8)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv4_num_out if i==1 else conv5_num_out,
                                        num_mid=num_mid,
                                        num_out=conv5_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        g=groups,
                                        first_block=(i==1))) for i in range(1,k_sec[5]+1)
                    ]))

        # final
        self.tail = nn.Sequential(OrderedDict([
                    ('bn', nn.BatchNorm3d(conv5_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))

        if dropout:
            self.globalpool = nn.Sequential(OrderedDict([
                            ('avg', nn.AvgPool3d(kernel_size=(8,7,7), stride=(1,1,1))),
                            ('dropout', nn.Dropout(p=dropout)),
                            ]))
        else:
            self.globalpool = nn.Sequential(OrderedDict([
                            ('avg', nn.AvgPool3d(kernel_size=(8,7,7),  stride=(1,1,1))),
                            # ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
                            ]))
        self.classifier = nn.Linear(conv5_num_out, num_classes)

        # Initialization
        xavier(_net=self)

    def forward(self, x):
        assert x.shape[2] == 16
        _activations = []
        _weights = []

        h = self.conv1(x)  # x224 -> x112
        _activations.append(h)
        _weights.append(self.conv1.conv.weight.data)

        h = self.maxpool(h)  # x112 ->  x56

        h, w, a = self.conv2(h)  # x56 ->  x56
        _weights = _weights + w
        _activations = _activations + a

        h, w, a = self.conv3(h)  # x56 ->  x28
        _weights = _weights + w
        _activations = _activations + a

        h, w, a = self.conv4(h)  # x28 ->  x14
        _weights = _weights + w
        _activations = _activations + a

        h, w, a = self.conv5(h)  # x14 ->   x7
        _weights = _weights + w
        _activations = _activations + a

        h = self.tail(h)
        # _activations.append(h)
        _activations[-1] = h
        h = self.globalpool(h)
        _activations.append(h)

        h = h.view(h.shape[0], -1)
        h = self.classifier(h)
        _weights.append(self.classifier.weight.data)

        return h, _weights, _activations


if __name__ == "__main__":
    import torch
    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = MFNET_3D(num_classes=125, pretrained=False)
    data = torch.tensor(torch.randn(1,3,16,224,224))
    output, weights, activations = net(data)

    print("")
