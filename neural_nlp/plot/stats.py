import numpy as np


def is_significant(scores, reference_scores, samples=10000, pvalue_threshold=0.05):
    delta = np.mean(scores) - np.mean(reference_scores)
    if delta < 0:
        return False  # reference scores are already higher, don't bother checking
    pooled = np.hstack([scores, reference_scores])
    estimates = np.array([_permutation_test(pooled, len(scores), len(reference_scores))
                          for _ in range(samples)])
    diff = len(np.where(estimates <= delta)[0])
    p = 1.0 - diff / samples
    return p < pvalue_threshold


def _permutation_test(pooled, size1, size2):
    np.random.shuffle(pooled)
    shuffle1, shuffle2 = pooled[:size1], pooled[-size2:]
    return shuffle1.mean() - shuffle2.mean()
