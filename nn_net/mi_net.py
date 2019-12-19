import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from nn_net.misc import *
from nn_net.encoder import *


class MIFCNet(nn.Module):
    """Simple custom network for computing MI.
    Residual linear net to compute MINE
    """
    def __init__(self, n_input, n_units, bn=False):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super(MIFCNet, self).__init__()

        self.bn = bn

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        """
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: network output.
        """
        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)

        return h


class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.
    """
    def __init__(self, n_input, n_units):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super(MI1x1ConvNet, self).__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """
            Args:
                x: Input tensor.
            Returns:
                torch.Tensor: network output.
        """

        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h


class TriLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TriLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, input_):
        output = F.relu(self.fc1(input_))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class flate_enc_IB(nn.Module):
    def __init__(self, device, x_shape, code_shape):
        """
        Replication of statistics network from MINE
        """
        super(flate_enc_IB, self).__init__()
        self.device = device

        try:
            input_dim = np.prod(x_shape) + code_shape[0]
        except:
            input_dim = x_shape + code_shape

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, 1)

    def forward(self, x, z):

        # print x.shape
        fx = x.flatten(start_dim=1)
        # print fx.shape

        inp = torch.cat((z, fx), 1)
        b = fx.shape[0]

        z_dim = z.shape[1]
        x_dim = fx.shape[1]
        l = z_dim + x_dim
        e_1 = np.random.normal(0., np.sqrt(0.3), [b, l])
        e_2 = np.random.normal(0., np.sqrt(0.5), [b, l])
        e_3 = np.random.normal(0., np.sqrt(0.5), [b, l])

        e_1 = torch.from_numpy(e_1).float().to(device=self.device)
        e_2 = torch.from_numpy(e_2).float().to(device=self.device)
        e_3 = torch.from_numpy(e_3).float().to(device=self.device)

        s = F.elu(self.fc1(inp + F.sigmoid(e_1)))
        s = F.elu(self.fc2(s + F.sigmoid(e_2)))
        s = self.fc3(s + F.sigmoid(e_3))
        return s


if __name__ == '__main__':
    y = torch.ones(1, 128, 8, 8)
    minet = MI1x1ConvNet(128, 1)
    yp = minet(y)
    print yp.shape

    minet1 = MIFCNet(8, 256)
    ypp = minet1(y)
    print ypp.shape
