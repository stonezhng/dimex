import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import MNIST
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


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


def main(test_loader, args, input_shape, num_labels):
    checkpoints = torch.load('results/' + args.exp_id + '/minmax_model.pth.tar')

    # encoder, output c are used as the clf, output z are used as the code
    E = encoder.ImgnEncoder(input_shape).to(device=device)

    shape_hist = E.shape_hist
    feature_shape = shape_hist[E.config['feature_idx']]
    code_shape = shape_hist[-1]

    E.load_state_dict(checkpoints['E_state_dict'])

    E.eval()
    gt = []
    images = []

    codes = []

    for batch_idx, (X, y) in enumerate(test_loader):
        X = X.view(-1, *input_shape).to(device=device)

        gt.append(y.to(device='cpu').data.numpy())
        images.append(X.to(device='cpu').data.numpy().reshape(-1, *input_shape))
        c, z = E(X)

        codes.append(c.to(device='cpu').data.numpy())

    codes = np.concatenate(codes, axis=0)
    print codes.shape
    np.save('results/' + args.exp_id + '/code.npy', codes)
    images = np.concatenate(images, axis=0)
    gt = np.concatenate(gt, axis=0)

    # cluster = AgglomerativeClustering(n_clusters=num_labels, affinity='euclidean', linkage='ward')
    # cluster.fit_predict(codes)
    # pred_labels = cluster.labels_

    kmeans = KMeans(n_clusters=num_labels, random_state=0).fit(codes)
    pred_labels = kmeans.labels_
    print pred_labels.shape
    print pred_labels

    print '#' * 20
    print 'error rate: ', summary.cluster_errrate(pred_labels, gt, num_labels)
    f_img = summary.cluster_imgs(images[:100], pred_labels[:100], num_labels)

    if len(f_img.shape) > 2 and f_img.shape[2] > 3:
        f_img = np.moveaxis(f_img, 0, -1)

    fig, _ = plt.subplots()
    plt.imshow(f_img)
    plt.savefig('results/' + args.exp_id + '/cluster.png')
    plt.close(fig)