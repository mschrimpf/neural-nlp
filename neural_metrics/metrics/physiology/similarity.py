import itertools

import numpy as np


def rdm(data):
    return 1 - rsa(data)


def rsa(data):
    _rdm = np.empty((data.shape[0], data.shape[0]))
    for i, j in itertools.combinations_with_replacement(list(range(data.shape[0])), 2):
        _rdm[i, j] = _rdm[j, i] = np.corrcoef(data[i], data[j])[0, 1]
    return _rdm
