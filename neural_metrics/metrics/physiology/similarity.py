import itertools

import numpy as np


def rdm(data):
    return 1 - rsa(data)


def rsa(data):
    _rdm = np.empty((data.shape[0], data.shape[0]))
    for i, j in itertools.combinations_with_replacement(list(range(data.shape[0])), 2):
        correlation = np.corrcoef(data[i], data[j])
        assert len(correlation.shape) == 2 and correlation.shape[0] == correlation.shape[1] == 2
        _rdm[i, j] = _rdm[j, i] = correlation[0, 1]
    return _rdm
