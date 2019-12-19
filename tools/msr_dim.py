
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
import tools.summary as summary

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


"""
Data loader
GPU/CPU check
"""
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", type=int, default=30)
parser.add_argument("--batches", "-b", type=int, default=100)
parser.add_argument("--lr", "-l", type=float, default=5.E-4)
parser.add_argument("--gamma", "-G", type=float, default=0.01)
parser.add_argument("--exp_id", "-I", type=str, default='msr0')
args = parser.parse_args()

if not os.path.exists('results/'+args.exp_id):
    os.makedirs('results/'+args.exp_id)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_devices = torch.cuda.device_count()

thousand = np.arange(100)

rX = np.load('r_X_sample_10000_t90.npy')[:, :-1]
freq = rX[:, -1][:, np.newaxis]

N, Fs = rX.shape

train =torch.utils.data.TensorDataset(torch.FloatTensor(rX).to(device))
train_loader = torch.utils.data.DataLoader(train, batch_size=args.batches, shuffle=True, num_workers=4 * num_devices)


def ma(a, window_size=100):
    """
    sliding window smoothness
    """
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_devices = torch.cuda.device_count()


def main(train_loader, args):

    """
    NN models
    """
    # encoder, output c are used as the clf, output z are used as the code
    E = encoder.LinearEnc(Fs, 20, 2).to(device=device)

    # "Discriminator", AKA the T net in MINE, for global and local MI estimation
    GD = mi_net.flate_enc_IB(device, Fs, 2).to(device=device)

    # This one is a true discriminator
    # PD = encoder.LinearEnc(2, 10, 1).to(device=device)
    """
    Hyperparams
    """
    EPOCH = args.epochs
    lr = float(args.lr)
    gamma = args.gamma

    """
    Optimizer
    """
    # opt = torch.optim.Adam(
    #     list(E.parameters()) + list(GD.parameters()) + list(PD.parameters()),
    #     lr=lr)
    opt = torch.optim.Adam(
        list(E.parameters()) + list(GD.parameters()) ,
        lr=lr)

    global_mi = []
    prior_match = []
    dim = []

    """
    Train
    """
    for e in range(EPOCH):
        E.train()
        GD.train()
        # PD.train()

        codes = []

        for batch_idx, (X,) in enumerate(train_loader):
            opt.zero_grad()

            X = X.view(-1, Fs).to(device=device)
            # print(X)

            X_cpu = X.to(device='cpu')
            X_hat = torch.from_numpy(np.random.permutation(X_cpu))
            X_hat = X_hat.to(device=device)

            # z: global feature (label), c: feature map
            z = E(X)
            g_mi = lb.biased_DV(GD, z, X, X_hat)

            # g_mi = lb.EB(GD, alpha_net,  c, z)
            # l_mi = torch.mean(lb.EB(LD, alpha_net, c, z))

            # prior match
            # p = torch.FloatTensor(*z.shape).uniform_(0, 1).to(device=device)
            # term_a = torch.mean(torch.log(PD(p)))
            # term_b = torch.mean(torch.log(1. - PD(z)))
            # pm = term_a + term_b

            # loss = -g_mi - gamma * pm
            loss = -g_mi

            loss.backward()
            opt.step()

            global_mi.append(g_mi.item())

            print ('epoch %d, dim %f' % (e, -loss.item()))

            dim.append(-loss.item())

            codes.append(z.cpu().data.numpy())

        gma = ma(global_mi)
        pma = ma(prior_match)

        fig, _ = plt.subplots()
        plt.xlabel('windows')
        plt.plot(pma, color='b', label='prior match')
        plt.plot(gma, color='g', label='global mi')
        plt.legend()
        plt.savefig('results/' + args.exp_id + '/dim_part.png')
        # plt.cla()
        plt.close(fig)

        dma = ma(dim)
        fig, _ = plt.subplots()
        plt.xlabel('windows')
        plt.plot(dma, color='b', label='ma curve')
        plt.legend()
        plt.savefig('results/' + args.exp_id + '/dim.png')
        # plt.cla()
        plt.close(fig)

        if (e+1) % 10 == 0:
            state = {
                'epoch': e + 1,
                'GD_state_dict': GD.state_dict(),
                # 'PD_state_dict': PD.state_dict(),
                'E_state_dict': E.state_dict(),
                'opt': opt.state_dict()
            }
            fh.save_checkpoint(state, filename='results/' + args.exp_id + '/minmax_model.pth.tar')

            """
            Test
            """
            print '#' * 20
            """
            Reload NN models
            """
            checkpoints = torch.load('results/' + args.exp_id + '/minmax_model.pth.tar')

            # encoder, output c are used as the clf, output z are used as the code
            E = encoder.LinearEnc(Fs, 20, 2).to(device=device)

            E.load_state_dict(checkpoints['E_state_dict'])

            E.eval()
            codes = []

            for batch_idx, (X,) in enumerate(train_loader):
                X = X.view(-1, Fs).to(device=device)
                z = E(X)
                codes.append(z.to(device='cpu').data.numpy())

            codes = np.concatenate(codes, axis=0)
            codes = np.concatenate((codes, freq), axis=1)
            np.save('rX_reduces.npy', codes)


if __name__ == '__main__':
    main(train_loader, args)
    # print(rX)
