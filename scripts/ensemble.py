import numpy as np
import torch

from .normalize import normalize
from sklearn.neural_network import BernoulliRBM

import matplotlib.pyplot as plt


def generate_ensembles(attributions, methods, device="cpu"):

    size = [len(methods)] + list(attributions.shape)[1:]
    e = torch.empty(size=size).to(device)

    for i, m in enumerate(methods):
        ens = ensemble(m, attributions=attributions)
        e[i] = ens

    return e


def ensemble(name, **kwargs):

    if name == "mean":
        return mean_ens(**kwargs)

    if name == "variance":
        return variance_ens(**kwargs)

    if name == "rbm":
        return rbm_ens(**kwargs)


def mean_ens(attributions):
    return torch.mean(attributions, dim=0)
    # return np.mean(attributions, axis=1)


def variance_ens(attributions):
    # TODO epsilon should be mean over the whole dataset, not just the batch
    epsilon = torch.mean(attributions) * 10
    return torch.mean(attributions, dim=0) / (torch.std(attributions, dim=0) + epsilon)


def rbm_ens(attributions):

    # TODO use parameters
    rbms = [BernoulliRBM(n_components=1, batch_size=10, learning_rate=0.01, n_iter=100)]

    A = attributions.clone()

    # change (attribution methods, batch) into (batch, attribution methods)
    A = torch.transpose(A, 0, 1)
    A = A.cpu().detach().numpy()

    # make last two dimensions (width * height) into one
    A = A.reshape(A.shape[0], A.shape[1], A.shape[2] * A.shape[3])

    size = list(attributions.shape)[1:]
    result = torch.empty(size=size)
    for i, a in enumerate(A):

        a = a.T

        for r in rbms:
            r.fit(a)
            # sample output from current rbm as input for the next rbm
            a = r.transform(a)

        result[i] = torch.tensor(a.reshape(size[-1], size[-1]))

    return result
