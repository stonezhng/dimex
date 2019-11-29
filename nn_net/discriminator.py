import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import nn_net.misc as misc
import nn_net.convnet as CNN

from copy import deepcopy


class ImgnLocalDisc(nn.Module):
    def __init__(self, feature_shape, code_shape):
        super(ImgnLocalDisc, self).__init__()
        cat_shape = (feature_shape[0] + code_shape[0], feature_shape[1], feature_shape[2])
        config = dict(
            input_shape=cat_shape,
            layers=[dict(layer='conv', args=(512, 1, 1, 0), bn=True, act='ReLU'),
                    dict(layer='conv', args=(512, 1, 1, 0), bn=True, act='ReLU'),
                    dict(layer='conv', args=(1, 1, 1, 0), bn=True, act='ReLU')],
            feature_idx=None
        )

        self.config = config
        self.conv = CNN.ConvNet(config)
        self.shape_hist = self.compute_shape()

    def compute_shape(self):
        shape = self.config['input_shape']
        shape_hist = []

        layers = self.config['layers']
        for l in layers:
            if l['layer'] == 'conv':
                dim_in, dim_x, dim_y = shape
                dim_out, f, s, p = l['args']
                shape = (dim_out, (dim_x - f + 2 * p) // s + 1, (dim_y - f + 2 * p) // s + 1)
                shape_hist.append(deepcopy(shape))
            elif l['layer'] == 'flatten':
                s = 1
                for d in shape:
                    s *= d
                shape = (s, )
                shape_hist.append(deepcopy(shape))
            elif l['layer'] == 'linear':
                shape = l['args']
                shape_hist.append(deepcopy(shape))
        return shape_hist

    def forward(self, Y, C):
        """
        :param C: feature map, (N, 256, 4, 4)
        :param Y: global feature vector, (N, 64)
        :return:
        """
        N, Gf = Y.shape
        _, Cf, h, w = C.shape

        ex_Y = Y.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        cat = torch.cat((ex_Y, C), dim=1)

        return self.conv(cat)  # (N, 1)


class ImgnGlobalDisc(nn.Module):
    def __init__(self, feature_shape, code_shape):
        super(ImgnGlobalDisc, self).__init__()
        conv2linear = {
            'input_shape': feature_shape,
            'layers': [
                # {'layer': 'conv', 'args': (64, 3, 1, 0), 'bn': False, 'act': 'ReLU'},
                # {'layer': 'conv', 'args': (32, 3, 1, 0), 'bn': False, 'act': 'ReLU'},
                {'layer': 'flatten'},
            ],
            'feature_idx': None
        }

        output_shape = (np.prod(feature_shape) + code_shape[0],)

        linear2scalar = {
            'input_shape': output_shape,
            'layers': [
                {'layer': 'linear', 'args': (512,), 'bn': False, 'act': 'ReLU'},
                {'layer': 'linear', 'args': (512,), 'bn': False, 'act': 'ReLU'},
                {'layer': 'linear', 'args': (1,), 'bn': False, 'act': None}
            ],
            'feature_idx': None
        }

        self.required_input_shape = conv2linear['input_shape']

        self.conv2linear = CNN.ConvNet(conv2linear)
        self.linear2scalar = CNN.ConvNet(linear2scalar)

    def forward(self, Y, C):
        """
        :param C: feature map, (N, 256, 4, 4)
        :param Y: global feature vector, (N, 64)
        :return:
        """
        h = self.conv2linear(C)
        h = torch.cat((h, Y), dim=1)
        return self.linear2scalar(h)


class ImgnPriorDisc(nn.Module):
    def __init__(self, input_shape):
        super(ImgnPriorDisc, self).__init__()

        config = dict(
            input_shape=input_shape,
            layers=[dict(layer='linear', args=(1000, ), bn=False, act='ReLU'),
                    dict(layer='linear', args=(200, ), bn=False, act='ReLU'),
                    dict(layer='linear', args=(1, ), bn=False, act='sigmoid')],
            feature_idx=None
        )
        self.config = config
        self.linear2scalar = CNN.ConvNet(config)

    def forward(self, X):
        """
        :param X: Uniformly generated (N, 64) noisy global feature vector
        :return:
        """
        return self.linear2scalar(X)
