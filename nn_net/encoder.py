import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import nn_net.misc as misc


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


class ConvNet(nn.Module):
    """
    Use config dict to initialize a conv net
    """
    _supported_types = ('linear', 'conv', 'flatten', None)

    def built_layers(self, description, input_shape):
        modules_seq = nn.Sequential()
        if description['layer'] == 'conv':
            dim_in, dim_x, dim_y = input_shape
            dim_out, f, s, p = description['args']
            conv = nn.Conv2d(dim_in, dim_out, kernel_size=f, stride=s, padding=p)
            nn.init.xavier_uniform_(conv.weight)
            modules_seq.add_module('conv', conv)
            out_shape = (dim_out, (dim_x - f + 2 * p) // s + 1, (dim_y - f + 2 * p) // s + 1)
            self.end_of_layer(modules_seq, description, out_shape)
        elif description['layer'] == 'linear':
            if len(input_shape) == 3:
                dim_in, dim_x, dim_y = input_shape
                shape = (dim_x * dim_y * dim_in,)
                modules_seq.add_module('flatten', misc.View(-1, shape[0]))
            else:
                shape = input_shape
            dim_in, = shape
            dim_out, = description['args']
            fc = nn.Linear(dim_in, dim_out)
            nn.init.xavier_uniform_(fc.weight)
            modules_seq.add_module('linear', fc)
            out_shape = (dim_out, )
            self.end_of_layer(modules_seq, description, out_shape)
        elif description['layer'] == 'flatten':
            if len(input_shape) == 3:
                dim_in, dim_x, dim_y = input_shape
                shape = (dim_x * dim_y * dim_in,)
            else:
                shape = input_shape
            modules_seq.add_module('flatten', misc.View(-1, shape[0]))
            out_shape = shape
        else:
            raise NotImplementedError(
                'Layer {} not supported. Use {}'.format(description['layer'], self._supported_types))

        return modules_seq, out_shape

    def end_of_layer(self, modules_seq, description, out_shape):
        if len(out_shape) == 3:
            dim_out, dim_x, dim_y = out_shape
            if description['bn']:
                modules_seq.add_module('bn', nn.BatchNorm2d(dim_out))
        else:
            if description['bn']:
                modules_seq.add_module('bn', nn.BatchNorm1d(out_shape[0]))
        if description['act'] == 'ReLU':
            modules_seq.add_module('activate', nn.ReLU())

    def __init__(self, configs):
        super(ConvNet, self).__init__()

        self.modules_seq = nn.Sequential()

        input_shape = configs['input_shape']
        layers = configs['layers']
        task_idx = configs['local_task_idx']

        out_shape = input_shape
        for i, description in enumerate(layers):
            modules_block, out_shape = self.built_layers(description, out_shape)
            self.modules_seq.add_module('layer{}'.format(i), modules_block)
        self.local_idx = task_idx[0]
        self.global_idx = task_idx[1]

    def forward(self, inputs, return_full_list=False):
        out = []
        for l in self.modules_seq:
            inputs = l(inputs)
            out.append(inputs)
        if return_full_list:
            return out
        elif self.local_idx is not None and self.global_idx is not None:
            return out[-1], out[self.local_idx], out[self.global_idx]
        else:
            return out[-1]

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