import numpy as np
import torch


def normalize(method, **kwargs):

    if method == "min_max":
        return min_max(**kwargs)

    # if method == "sum":
    #     return normalize_sum1(**kwargs)


def min_max(arr, arr2=None):
    if arr2 is None:
        # Do min-max normalization for just one array
        A = [arr.clone()]
    else:
        # Do min max normalization for both arrays
        # The difference later is the division by max distance
        # of the two arrays
        A = [arr.clone(), arr2.clone()]

    A = torch.stack(A)
    A = A.view(-1, arr.size(0), arr.size(1), arr.size(2)*arr.size(3))
    A -= A.min(-1, keepdim=True)[0]
    # If arr2<> None, divide b
    A /= A.amax((0, -1), keepdim=True)
    A = A.view([-1] + list(arr.size()))
    if arr2 is None:
        return A[0]
    else:
        return A[0], A[1]