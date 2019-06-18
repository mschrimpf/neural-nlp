import numpy as np


def ordered_set(l):
    if isinstance(l, np.ndarray):
        l = l.tolist()
    return sorted(set(l), key=l.index)


def is_sorted(x):
    return all(x[i] <= x[i + 1] for i in range(len(x) - 1))
