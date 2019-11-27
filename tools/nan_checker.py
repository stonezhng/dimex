import torch


def contains_nan(S):
    return torch.sum(torch.isnan(S)) > 0


def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print name, param.data