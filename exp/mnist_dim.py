import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
import matplotlib.pyplot as plt

import numpy as np

import nn_net.mi_net as mi_net
import nn_net.encoder as encoder
import estimator.lower_bound as lb
import tools.file_helper as fh

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import os


def ma(a, window_size=100):
    """
    sliding window smoothness
    """
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


"""
Data loader
GPU/CPU check
"""
parser = argparse.ArgumentParser()
parser.add_argument("epochs", type=int)
parser.add_argument("batches", type=int)
parser.add_argument("debug")
parser.add_argument("exp-id")

args = parser.parse_args()
args = vars(args)

if not os.path.exists('results/'+args['exp-id']):
    os.makedirs('results/'+args['exp-id'])

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_devices = torch.cuda.device_count()

thousand = np.arange(1000)

mnist_train = datasets.MNIST(root='.', train=True, download=True,
                             transform=transforms.Compose(
                                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                             ))

mnist_test = datasets.MNIST(root='.', train=False, download=True,
                            transform=transforms.Compose(
                                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                            ))

train_subset = torch.utils.data.Subset(mnist_train, thousand)
test_subset = torch.utils.data.Subset(mnist_test, thousand)

if args['debug'] == 'debug':
    trainset = train_subset
    testset = test_subset
else:
    trainset = mnist_train
    testset = test_subset

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args['batches'], shuffle=True, num_workers=4 * num_devices)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args['batches'], shuffle=True, num_workers=4 * num_devices)
