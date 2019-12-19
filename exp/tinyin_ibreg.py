import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
import matplotlib.pyplot as plt

import numpy as np

import nn_net.mi_net as mi_net
import nn_net.encoder as encoder
import nn_net.discriminator as discriminator
import estimator.lower_bound as lb
import tools.file_helper as fh

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import os

import exp.dim_ibreg as dimib
import exp.dim_ibreg_test as dimib_test

"""
Data loader
GPU/CPU check
"""
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", type=int, default=10)
parser.add_argument("--batches", "-b", type=int, default=64)
parser.add_argument("--lr", "-l", type=float, default=1.E-4)
parser.add_argument("--alpha", "-A", type=float, default=0.5)
parser.add_argument("--beta", "-B", type=float, default=1.0)
parser.add_argument("--gamma", "-G", type=float, default=0.1)
parser.add_argument("--exp_id", "-I", type=str, default='tiny0')
parser.add_argument("--debug", "-D", type=int, default=1)
parser.add_argument("--test", "-T", type=int, default=0)
args = parser.parse_args()

if not os.path.exists('results/'+args.exp_id):
    os.makedirs('results/'+args.exp_id)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_devices = torch.cuda.device_count()

thousand = np.arange(100)

X_train = np.load('X_train_tinyin.npy')
Y_train = np.load('Y_train_tinyin.npy')
X_train = np.swapaxes(X_train, 1, 3)

X_val = np.load('X_val_tinyin.npy')
Y_val = np.load('Y_val_tinyin.npy')
X_val = np.swapaxes(X_val, 1, 3)

X_subtrain = X_train[Y_train < 10].copy()
Y_subtrain = Y_train[Y_train < 10].copy()

X_subval = X_val[Y_val < 10].copy()
Y_subval = Y_val[Y_val < 10].copy()

print X_subtrain.shape
print Y_subtrain.shape
print X_subval.shape
print Y_subval.shape

mnist_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                             torch.from_numpy(Y_train).float().to(device))

mnist_test = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float().to(device),
                                            torch.from_numpy(Y_val).float().to(device))

sub_train = torch.utils.data.TensorDataset(torch.from_numpy(X_subtrain).float().to(device),
                                             torch.from_numpy(Y_subtrain).float().to(device))

sub_val = torch.utils.data.TensorDataset(torch.from_numpy(X_subval).float().to(device),
                                            torch.from_numpy(Y_subval).float().to(device))

train_subset = torch.utils.data.Subset(mnist_train, thousand)
test_subset = torch.utils.data.Subset(mnist_test, thousand)

print args.debug

if args.debug:
    trainset = train_subset
    testset = test_subset
else:
    trainset = mnist_train
    testset = mnist_test
    # trainset = sub_train
    # testset = sub_val

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batches, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batches, shuffle=True, num_workers=0)

if not args.test:
    dimib.main(train_loader, test_loader, args, (3, 64, 64), 10)
else:
    dimib_test.main(test_loader, args, (3, 64, 64), 10)