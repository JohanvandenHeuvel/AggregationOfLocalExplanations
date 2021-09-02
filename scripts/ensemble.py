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
        elif m == "rbm_flip_detection":
            # Requires all three calculated before
            rbm_index = methods.index("rbm")
            flipped_index = methods.index("flipped_rbm")
            baseline_index = methods.index("mean")

            e[i] = solve_flipping(e[rbm_index], e[flipped_index], e[baseline_index])
        else:
            ens = ensemble(m, attributions=attributions)
            e[i] = ens

    return e


def solve_flipping(rbm, rbm_flipped, baseline):
    # Define percentage of top baseline pixels that are compared
    pct_pixel_to_check = 5
    nr_pixels = rbm.shape[1] * rbm.shape[2]
    barrier = int(pct_pixel_to_check/100 * nr_pixels)

    # Get most important pixel positions of baseline
    compare_pixels = torch.argsort(baseline.reshape(-1, nr_pixels), dim=1)[:, -barrier:]

    # Sort rbm pixels by relevancy
    rbm_rank = torch.argsort(rbm.reshape(-1, nr_pixels), dim=1)

    # Compute how many of the top baseline pixels are
    # most relevant / least relevant pixels for the rbm using the percentage of pixels
    rbm_best = calc_count_intersects(compare_pixels, rbm_rank[:, -barrier:])
    rbm_worst = calc_count_intersects(compare_pixels, rbm_rank[:, :barrier])

    # Replace the original rbm by the flipped version if worst scores outweight best scores
    solved_rbm = rbm.clone()
    replace_index = np.asarray(rbm_best) - np.asarray(rbm_worst) < 0
    solved_rbm[replace_index] = rbm_flipped[replace_index]

    return solved_rbm


def calc_count_intersects(t1, t2):
    # Calculate the number of elements contained in set t1 and set t2
    # Dimension 0 is the batch dimension

    # Combine the tensors, so that later we can find duplicates
    combined = torch.cat((t1, t2), dim=1)

    # Identify the duplicates of the combined set
    # Unfortuntely batches don't work with unique function
    c = [combined[i].unique(return_counts=True)[1] for i in range(combined.shape[0])]

    # Count the duplicates
    count_intersect = [torch.sum(c[i]>1).item() for i in range(len(c))]

    return count_intersect


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
