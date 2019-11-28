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
parser.add_argument("--epochs", "-e", type=int, default=100)
parser.add_argument("--batches", "-b", type=int, default=64)
parser.add_argument("--lr", "-l", type=float, default=1.E-4)
parser.add_argument("--alpha", "-A", type=float, default=0.5)
parser.add_argument("--beta", "-B", type=float, default=1.0)
parser.add_argument("--gamma", "-G", type=float, default=0.1)
parser.add_argument("--exp_id", "-I", type=int)
parser.add_argument("--debug", "-D", type=bool, default=False)
args = parser.parse_args()

if not os.path.exists('results/'+args.exp_id):
    os.makedirs('results/'+args.exp_id)

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

if args.debug:
    trainset = train_subset
    testset = test_subset
else:
    trainset = mnist_train
    testset = test_subset

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batches, shuffle=True, num_workers=4 * num_devices)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batches, shuffle=True, num_workers=4 * num_devices)

"""
NN models
"""
# encoder, output c are used as the clf, output z are used as the code
E = encoder.ImgnEncoder((1, 28, 28)).to(device=device)

shape_hist = E.shape_hist
feature_shape = shape_hist[E.config['feature_idx']]
code_shape = shape_hist[-1]

# "Discriminator", AKA the T net in MINE, for global and local MI estimation
GD = discriminator.ImgnGlobalDisc(feature_shape, code_shape).to(device=device)
LD = discriminator.ImgnLocalDisc(feature_shape, code_shape).to(device=device)

# This one is a true discriminator
PD = discriminator.ImgnPriorDisc(code_shape).to(device)

"""
Hyperparams
"""
EPOCH = args.epochs
lr = float(args.lr)
alpha = args.alpha
beta = args.beta
gamma = args.gamma

"""
Optimizer
"""
opt = torch.optim.Adam(
    [E.parameters(), GD.parameters(), LD.parameters(), PD.parameters()],
    lr=lr)

local_mi = []
global_mi = []
prior_match = []
dim = []

clf_loss = []

min_loss = 1e8


"""
Train
"""
for e in range(args['epochs']):
    E.train()
    GD.train()
    LD.train()
    PD.train()

    for batch_idx, (X, y) in enumerate(train_loader):
        opt.zero_grad()

        X = X.view(-1, 1, 28, 28).to(device=device)
        y = y.to(device=device)

        X_cpu = X.to(device='cpu')
        X_hat = torch.from_numpy(np.random.permutation(X_cpu))
        X_hat = X_hat.to(device=device)

        # z: global feature (label), c: feature map
        z, c = E(X)
        _, c_hat = E(X_hat)
        g_mi = lb.biased_DV(GD, z, c, c_hat)
        l_mi = lb.biased_DV(LD, z, c, c_hat)

        # prior match
        p = torch.FloatTensor(*z.shape).uniform_(0, 1)
        term_a = torch.mean(torch.log(PD(p)))
        term_b = torch.mean(torch.log(1. - PD(z)))
        pm = term_a + term_b

        loss = -alpha * g_mi - beta * l_mi - gamma * pm

        loss.backward()
        opt.step()

        global_mi.append(g_mi.item())
        local_mi.append(l_mi.item())
        prior_match.append(pm.data.numpy())

        clf = F.cross_entropy(z, y)

        print ('epoch %d, loss %f, clf loss %f' % (e, loss.item(), clf.item()))

        clf_loss.append(clf.item())
        dim.append(-loss.item())

    gma = ma(global_mi)
    lma = ma(local_mi)
    pma = ma(prior_match)

    fig, _ = plt.subplots()
    plt.xlabel('epoch')
    plt.legend()
    plt.plot(pma, color='b', label='prior match')
    plt.plot(pma, color='r', label='local mi')
    plt.plot(pma, color='g', label='global mi')
    plt.savefig('results/' + args['exp-id'] + '/dim_part.png')
    # plt.cla()
    plt.close(fig)

    dma = ma(dim)
    fig, _ = plt.subplots()
    plt.xlabel('epoch')
    plt.legend()
    plt.plot(dma, color='b', label='ma curve')
    plt.plot(dim, color='r', label='raw curve')
    plt.savefig('results/' + args['exp-id'] + '/dim.png')
    # plt.cla()
    plt.close(fig)

    fig, _ = plt.subplots()
    plt.xlabel('epoch')
    plt.plot(clf_loss, color='b')
    plt.savefig('results/' + args['exp-id'] + '/clf.png')
    # plt.cla()
    plt.close(fig)


state = {
    'epoch': args['epochs'] + 1,
    'GD_state_dict': GD.state_dict(),
    'LD_state_dict': LD.state_dict(),
    'PD_state_dict': PD.state_dict(),
    'E_state_dict': E.state_dict(),
    'opt': opt.state_dict()
}
fh.save_checkpoint(state, filename='results/'+args['exp-id'] + '/last_minmax_model.pth.tar')
