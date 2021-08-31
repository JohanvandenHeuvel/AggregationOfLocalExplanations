import numpy as np
import torch


def normalize(method, **kwargs):

    if method == "min_max":
        return min_max(**kwargs)

    # if method == "duplex_absolute":
    #     return duplex_normalize_abs(**kwargs)
    #
    # if method == "sum":
    #     return normalize_sum1(**kwargs)


def min_max(arr):

    A = arr.clone()

    A = A.view(arr.size(0), arr.size(1), -1)
    A -= A.min(-1, keepdim=True)[0]
    A /= A.max(-1, keepdim=True)[0]
    A = A.view(arr.shape)

    return A


# def duplex_normalize_abs(arr1, arr2):
#     value1 = arr1 - np.min(arr1)
#     value2 = arr2 - np.min(arr2)
#
#     distance1 = np.max(arr1) - np.min(arr1)
#     distance2 = np.max(arr1) - np.min(arr1)
#     distance = max(distance1, distance2)
#
#     if distance > 0:
#         value1 = value1 / distance
#         value2 = value2 / distance
#
#     return value1, value2


# def normalize_sum1(arr, axis=None):
#     a = arr - np.min(arr, axis=axis)
#     if np.sum(a) != 0:
#         a = a / np.sum(a)
#     return a
