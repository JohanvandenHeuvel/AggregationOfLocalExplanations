import numpy as np


def normalize_abs(arr):
    value = arr - np.min(arr)

    distance = (np.max(arr) - np.min(arr))
    if distance > 0:
        value = value / distance

    return value


def duplex_normalize_abs(arr1, arr2):
    value1 = arr1 - np.min(arr1)
    value2 = arr2 - np.min(arr2)

    distance1 = (np.max(arr1) - np.min(arr1))
    distance2 = (np.max(arr1) - np.min(arr1))
    distance = max(distance1, distance2)


    if distance > 0:
        value1 = value1 / distance
        value2 = value2 / distance

    return value1, value2


def normalize_sum1(arr, axis=None):
    a = arr - np.min(arr, axis=axis)
    if np.sum(a) != 0:
        a = a / np.sum(a)
    return a
