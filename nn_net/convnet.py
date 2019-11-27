import torch
import torch.nn as nn

import nn_net.misc as misc


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
        self.feature_idx = configs['feature_idx']

        out_shape = input_shape
        for i, description in enumerate(layers):
            modules_block, out_shape = self.built_layers(description, out_shape)
            self.modules_seq.add_module('layer{}'.format(i), modules_block)
        # self.feature_idx = task_idx[0]
        # self.global_idx = task_idx[1]

    def forward(self, inputs, return_full_list=False):
        out = []
        for l in self.modules_seq:
            inputs = l(inputs)
            out.append(inputs)
        if return_full_list:
            return out
        elif self.feature_idx is not None:
            return out[-1], out[self.feature_idx]
        else:
            return out[-1]


if __name__ == '__main__':
    import configs.encoder_config as config
    test_encoder = ConvNet(config.basic32x32)
    x = torch.ones(1, 3, 32, 32)
    ey, ly, gy = test_encoder(x)
    print(ey.shape)
    print(gy.shape)
    print(ly.shape)
