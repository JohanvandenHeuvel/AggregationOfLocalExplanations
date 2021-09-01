import numpy as np
import torch


def normalize(method, **kwargs):

    if method == "min_max":
        return min_max(**kwargs)

    # if method == "sum":
    #     return normalize_sum1(**kwargs)


def min_max(arr):

    A = arr.clone()

    A = A.view(arr.size(0), arr.size(1), -1)
    A -= A.min(-1, keepdim=True)[0]
    A /= A.max(-1, keepdim=True)[0]
    A = A.view(arr.shape)

    return A
