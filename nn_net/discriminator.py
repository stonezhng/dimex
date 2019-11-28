import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import nn_net.misc as misc
import nn_net.convnet as CNN


class ImgnLocalDisc(nn.Module):
    def __init__(self, input_shape):
        super(ImgnLocalDisc, self).__init__()

        config = dict(
            input_shape=input_shape,
            layers=[dict(layer='conv', args=(512, 1, 1, 0), bn=True, act='ReLU'),
                    dict(layer='conv', args=(512, 1, 1, 0), bn=True, act='ReLU'),
                    dict(layer='conv', args=(1, 1, 1, 0), bn=True, act='ReLU')],
            feature_idx=None
        )

        self.conv = CNN.ConvNet(config)

    def forward(self, C, Y):
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
    def __init__(self, input_shape):
        super(ImgnGlobalDisc, self).__init__()
        conv2linear = {
            'input_shape': input_shape,
            'layers': [
                # {'layer': 'conv', 'args': (64, 3, 1, 0), 'bn': False, 'act': 'ReLU'},
                # {'layer': 'conv', 'args': (32, 3, 1, 0), 'bn': False, 'act': 'ReLU'},
                {'layer': 'flatten'},
            ],
            'local_task_idx': (None, None)
        }

        output_shape = (sum(input_shape) + 64, )

        linear2scalar = {
            'input_shape': output_shape,
            'layers': [
                {'layer': 'linear', 'args': (512,), 'bn': False, 'act': 'ReLU'},
                {'layer': 'linear', 'args': (512,), 'bn': False, 'act': 'ReLU'},
                {'layer': 'linear', 'args': (1,), 'bn': False, 'act': None}
            ],
            'local_task_idx': (None, None)
        }

        self.required_input_shape = conv2linear['input_shape']

        self.conv2linear = CNN.ConvNet(conv2linear)
        self.linear2scalar = CNN.ConvNet(linear2scalar)

    def forward(self, C, Y):
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
        self.linear2scalar = CNN.ConvNet(config)

    def forward(self, X):
        """
        :param X: Uniformly generated (N, 64) noisy global feature vector
        :return:
        """
        return self.linear2scalar(X)
