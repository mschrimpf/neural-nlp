import numpy as np
import scipy.stats


def pearsonr_matrix(data1, data2, axis=1):
    rs = []
    for i in range(data1.shape[axis]):
        d1 = np.take(data1, i, axis=axis)
        d2 = np.take(data2, i, axis=axis)
        r, p = scipy.stats.pearsonr(d1, d2)
        rs.append(r)
    return np.array(rs)