import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import nn_net.misc as misc
import nn_net.convnet as CNN

from copy import deepcopy


class LinearEnc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearEnc, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_):
        output = F.relu(self.fc1(input_))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class enc_clf(nn.Module):
    def __init__(self):
        """
        Replication of Permutation Invariant MNIST FROM MINE
        """
        super(enc_clf, self).__init__()

        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        z = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        c = self.fc4(F.relu(z))
        return c, z


class ImgnEncoder(nn.Module):
    def __init__(self, input_shape):
        super(ImgnEncoder, self).__init__()

        self.config = dict(
            input_shape=input_shape,
            layers=[dict(layer='conv', args=(64, 4, 2, 1), bn=True, act='ReLU'),
                    dict(layer='conv', args=(128, 4, 2, 1), bn=True, act='ReLU'),
                    dict(layer='conv', args=(256, 4, 2, 1), bn=True, act='ReLU'),
                    dict(layer='conv', args=(512, 4, 2, 1), bn=True, act='ReLU'),
                    dict(layer='flatten'),
                    dict(layer='linear', args=(64,), bn=True, act='ReLU')],
            feature_idx=1
        )
        self.conv = CNN.ConvNet(self.config)
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

    def forward(self, input_):
        return self.conv(input_)  # Y (global feature), C (feature map)

# class imgn_encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         with self.init_scope():
#             self.c0 = L.Convolution2D(None, 64, 4)
#             self.c1 = L.Convolution2D(64, 128, 4)
#             self.c2 = L.Convolution2D(128, 256, 4)
#             self.c3 = L.Convolution2D(256, 512, 4)
#             self.linear = L.Linear(None, 64)
#             #self.bn0 = L.BatchNormalization(64)
#             self.bn1 = L.BatchNormalization(128)
#             self.bn2 = L.BatchNormalization(256)
#             self.bn3 = L.BatchNormalization(512)
#
#     def __call__(self, x):
#         h = F.relu(self.c0(x))
#         features = F.relu(self.bn1(self.c1(h)))
#         h = F.relu(self.bn2(self.c2(features)))
#         h = F.relu(self.bn3(self.c3(h)))
#         return self.linear(h), features