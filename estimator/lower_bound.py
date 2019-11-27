import torch
import torch.utils.data as td
import numpy as np
import tools.nan_checker as nc


def unbiased_avg_DV(T, joint_cat, margin_cat, ma_et, ma_rate):
    """
    :param T: neural estimator
    :param x: (B, T) vec
    :param y: (B, T) vec
    :param yp: (B, T) vec
    :return: estimated mutual information, scalar
    """
    t = T(joint_cat)
    tp = T(margin_cat)
    # mi = torch.mean(t) - (torch.logsumexp(tp, 0) - torch.log(torch.cuda.FloatTensor(joint_cat.shape[0]))[0])
    et = torch.exp(tp)

    mi = torch.mean(t) - torch.log(torch.mean(et))

    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))

    return mi, loss, ma_et


def biased_DV(T, X, z, z_hat):
    t = T(X, z)  # (N, F) -> (N, 1)
    tp = T(X, z_hat)

    assert not nc.contains_nan(t)
    assert not nc.contains_nan(tp)
    assert not nc.contains_nan(torch.mean(t))
    assert not nc.contains_nan(torch.logsumexp(tp, 0))

    N = X.shape[0]
    # print (torch.mean(t), torch.logsumexp(tp, 0)[0], torch.log(torch.FloatTensor([N]))[0])

    return torch.mean(t) - (torch.logsumexp(tp, 0)[0] - torch.log(torch.FloatTensor([N]))[0])