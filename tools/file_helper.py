import torch
import os
import matplotlib.pyplot as plt
import glob

import numpy as np
from tqdm import tqdm

import cv2
import cPickle as pickle


def save_checkpoint(state, filename='../out/model/checkpoint.pth.tar'):
    torch.save(state, filename)


def tiny_imagenet_train(root_path):
    labels = [name for name in os.listdir('{}/train'.format(root_path)) if os.path.isdir(name)]
    label_map = {}
    X = []
    Y = []
    for idx, l in tqdm(enumerate(labels)):
        label_map[l] = idx

        image_path = "{}/train/{}/images/*.JPEG".format(root_path, l)
        image_path_list = glob.glob(image_path)
        for p in image_path_list:
            im = cv2.imread(p).astype(float)
            if np.max(im) > 1:
                im /= 255.0
            X.append(im)
            Y.append(idx)

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    print X.shape
    print Y.shape

    np.save('X_train_tinyin.npy', X)
    np.save('Y_train_tinyin.npy', Y)

    pickle.dump(label_map, open('../val/label_map.pkl', 'wb'))


def tiny_imagenet_val(root_path):
    label_map = pickle.load(open('label_map.pkl', 'rb'))
    val_labels = {}
    X = []
    Y = []
    with open('val_annotations.txt') as fp:
        for line in fp:
            arr = line.split('\t')
            val_labels[arr[0]] = label_map[arr[1]]

    image_path = "{}/val/images/*.JPEG".format(root_path)
    image_path_list = glob.glob(image_path)
    for p in image_path_list:
        name = p.split('/')[-1]
        Y.append(val_labels[name])
        im = cv2.imread(p).astype(float)
        if np.max(im) > 1:
            im /= 255.0
        X.append(im)

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    print X.shape
    print Y.shape

    np.save('X_val_tinyin.npy', X)
    np.save('Y_val_tinyin.npy', Y)


if __name__ == '__main__':
    # tiny_imagenet_train('/home/sidongzhang/data/tiny-imagenet-200')
    tiny_imagenet_val('/home/sidongzhang/data/tiny-imagenet-200')