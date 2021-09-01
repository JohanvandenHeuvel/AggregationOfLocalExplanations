import numpy as np
import torch

from .normalize import normalize
from sklearn.neural_network import BernoulliRBM

import matplotlib.pyplot as plt


def generate_ensembles(attributions, methods, device="cpu"):

    size = [len(methods)] + list(attributions.shape)[1:]
    e = torch.empty(size=size).to(device)

    for i, m in enumerate(methods):

        # TOOO temporary solution
        if m == "flipped_rbm":
            # TODO flip by 1- or 1/
            # add flipped rbm
            j = methods.index("rbm")
            e[i] = 1 / e[j]
        else:
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


# def compute_ensembles(attributions, noise_attributions, tasks, positive_filter):
#     width, height = attributions.shape[1:]
#     attributions = attributions.reshape(len(attributions), -1).T
#
#     if noise_attributions is not None:
#         noise_attributions = np.asarray(noise_attributions)
#         noise_attributions = noise_attributions.reshape(len(noise_attributions), -1)
#
#     ensemble_attributions = []
#     for task in tasks:
#         attr = attributions.copy()
#         if task["nr_noise"] > 0:
#             attr = np.concatenate((attr, noise_attributions[0 : task["nr_noise"]]))
#
#         if task["technique"] == "rbm" and not positive_filter:
#             pos_attr, neg_attr = attr.copy(), attr.copy()
#             pos_attr[pos_attr < 0] = 0
#             neg_attr[neg_attr > 0] = 0
#
#             attributions = np.array(
#                 [
#                     normalize("duplex_absolute", arr1=pos_attr[i], arr2=neg_attr[i])
#                     for i in range(len(pos_attr))
#                 ]
#             )
#             pos_attr, neg_attr = (attributions[:, 0], attributions[:, 1])
#
#             pos_ensemble = rbm_ens(task["rbm"], pos_attr)
#             neg_ensemble = rbm_ens(task["rbm"], neg_attr)
#
#             ensemble = pos_ensemble - neg_ensemble
#         else:
#             if positive_filter:
#                 attr[attr < 0] = 0
#             attr = np.array([normalize("absolute", arr=a) for a in attr])
#
#         if task["technique"] == "mean":
#             ensemble = mean_ens(attr)
#         elif task["technique"] == "var":
#             ensemble = variance_ens(attr)
#         elif task["technique"] == "rbm" and positive_filter:
#             ensemble = rbm_ens(task["rbm"], attr)
#         elif task["technique"] == "rbm" and not positive_filter:
#             pass  # Case captured above
#         else:
#             raise ValueError("Ensemble technique not found")
#
#         ensemble = ensemble.reshape(1, width, height)
#         ensemble_attributions.append(ensemble)
#
#     return np.stack(ensemble_attributions)


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
