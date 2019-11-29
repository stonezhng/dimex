import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
import matplotlib.pyplot as plt

import numpy as np

import nn_net.encoder as encoder
import nn_net.discriminator as discriminator
import nn_net.mi_net as mi_net

import estimator.lower_bound as lb
import tools.file_helper as fh
import tools.summary as summary

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import os

from sklearn.cluster import AgglomerativeClustering


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


def main(train_loader, test_loader, args, input_shape, num_labels):
    """
    NN models
    """
    # encoder, output c are used as the clf, output z are used as the code
    E = encoder.ImgnEncoder(input_shape).to(device=device)

    shape_hist = E.shape_hist
    feature_shape = shape_hist[E.config['feature_idx']]
    code_shape = shape_hist[-1]

    # "Discriminator", AKA the T net in MINE, for global and local MI estimation
    GD = discriminator.ImgnGlobalDisc(feature_shape, code_shape).to(device=device)
    LD = discriminator.ImgnLocalDisc(feature_shape, code_shape).to(device=device)

    # This one is a true discriminator
    PD = discriminator.ImgnPriorDisc(code_shape).to(device=device)

    # T for IB Regularizer
    RegT = mi_net.flate_enc_IB(device, input_shape, code_shape).to(device=device)

    """
    Hyperparams
    """
    EPOCH = args.epochs
    lr = float(args.lr)
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    lambda_ = args.lam

    """
    Optimizer
    """
    opt = torch.optim.Adam(
        list(E.parameters()) + list(GD.parameters()) + list(LD.parameters()) +
        list(PD.parameters()),
        lr=lr)

    ib_opt = torch.optim.Adam(RegT.parameters(), lr=lr)

    local_mi = []
    global_mi = []
    prior_match = []
    dim = []

    """
    Train
    """
    for e in range(EPOCH):
        E.train()
        GD.train()
        LD.train()
        PD.train()

        for batch_idx, (X, y) in enumerate(train_loader):
            opt.zero_grad()

            X = X.view(-1, *input_shape).to(device=device)
            y = y.to(device=device)

            X_cpu = X.to(device='cpu')
            X_hat = torch.from_numpy(np.random.permutation(X_cpu))
            X_hat = X_hat.to(device=device)

            # z: global feature (label), c: feature map
            z, c = E(X)
            z_hat, c_hat = E(X_hat)
            g_mi = lb.biased_DV(GD, z, c, c_hat)
            l_mi = torch.mean(lb.biased_DV(LD, z, c, c_hat))

            # prior match
            p = torch.FloatTensor(*z.shape).uniform_(0, 1)
            term_a = torch.mean(torch.log(PD(p)))
            term_b = torch.mean(torch.log(1. - PD(z)))
            pm = term_a + term_b

            # regularier
            mi_xz_loss = -lb.biased_DV(RegT, X, z, z_hat)

            mi_xz_loss.backward(retain_graph=True)
            ib_opt.step()

            loss = -alpha * g_mi - beta * l_mi - gamma * pm + lambda_ * (-mi_xz_loss)

            loss.backward()
            opt.step()

            global_mi.append(g_mi.item())
            local_mi.append(l_mi.item())
            prior_match.append(pm.data.numpy())

            print ('epoch %d, dim %f' % (e, -loss.item()))

            dim.append(-loss.item())

        gma = ma(global_mi)
        lma = ma(local_mi)
        pma = ma(prior_match)

        fig, _ = plt.subplots()
        plt.xlabel('epoch')
        plt.plot(pma, color='b', label='prior match')
        plt.plot(lma, color='r', label='local mi')
        plt.plot(gma, color='g', label='global mi')
        plt.legend()
        plt.savefig('results/' + args.exp_id + '/dim_part.png')
        # plt.cla()
        plt.close(fig)

        dma = ma(dim)
        fig, _ = plt.subplots()
        plt.xlabel('epoch')
        plt.plot(dma, color='b', label='ma curve')
        plt.plot(dim, color='r', label='raw curve')
        plt.legend()
        plt.savefig('results/' + args.exp_id + '/dim.png')
        # plt.cla()
        plt.close(fig)

    state = {
        'epoch': EPOCH + 1,
        'GD_state_dict': GD.state_dict(),
        'LD_state_dict': LD.state_dict(),
        'PD_state_dict': PD.state_dict(),
        'E_state_dict': E.state_dict(),
        'opt': opt.state_dict()
    }
    fh.save_checkpoint(state, filename='results/' + args.exp_id + '/last_minmax_model.pth.tar')

    """
    Test
    """
    print '#'*20
    """
    Reload NN models
    """
    checkpoints = torch.load('results/' + args.exp_id + '/last_minmax_model.pth.tar')

    # encoder, output c are used as the clf, output z are used as the code
    E = encoder.ImgnEncoder(input_shape).to(device=device)

    shape_hist = E.shape_hist
    feature_shape = shape_hist[E.config['feature_idx']]
    code_shape = shape_hist[-1]

    E.load_state_dict(checkpoints['E_state_dict'])

    E.eval()
    codes = []
    gt = []
    images = []

    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.view(-1, *input_shape).to(device=device)
        gt.append(y.to(device='cpu').data.numpy())
        images.append(X.to(device='cpu').data.numpy().reshape(-1, *input_shape))
        c, z = E(X)
        codes.append(c.to(device='cpu').data.numpy())

    codes = np.concatenate(codes, axis=0)
    images = np.concatenate(images, axis=0)
    gt = np.concatenate(gt, axis=0)

    cluster = AgglomerativeClustering(n_clusters=num_labels, affinity='euclidean', linkage='ward')
    cluster.fit_predict(np.array(codes))
    pred_labels = cluster.labels_

    print '#'*20
    print 'error rate: ', summary.cluster_errrate(pred_labels, gt, num_labels)
    f_img = summary.cluster_imgs(images, pred_labels, num_labels)

    fig, _ = plt.subplots()
    plt.imshow(f_img)
    plt.savefig('results/' + args.exp_id + '/cluster.png')
    plt.close(fig)
