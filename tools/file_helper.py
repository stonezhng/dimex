import torch


def save_checkpoint(state, filename='../out/model/checkpoint.pth.tar'):
    torch.save(state, filename)