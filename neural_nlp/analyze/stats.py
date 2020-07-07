import logging
import numpy as np
from scipy.stats import ttest_ind

from neural_nlp import score

logger = logging.getLogger(__name__)


def is_significant(scores, reference_scores, samples=10000, pvalue_threshold=0.05):
    delta = np.mean(scores) - np.mean(reference_scores)
    print('Delta: ' + str(delta))
    pooled = np.hstack([scores, reference_scores])
    estimates = np.array([_permutation_test(pooled, len(scores), len(reference_scores))
                          for _ in range(samples)])
    print('Shuffled estimates: ', str(np.mean(estimates)))
    diff = len(np.where(estimates <= delta)[0])
    p = 1.0 - diff / samples
    return delta, np.mean(estimates), p


def _permutation_test(pooled, size1, size2):
    np.random.shuffle(pooled)
    shuffle1, shuffle2 = pooled[:size1], pooled[-size2:]
    return shuffle1.mean() - shuffle2.mean()


def model_training_diff(model='glove', benchmark='Pereira2018-encoding'):
    trained = score(benchmark=benchmark, model=model)
    untrained = score(benchmark=benchmark, model=f'{model}-untrained')
    # get per-subject scores. do not need to worry about ceiling because it is identical due to same benchmark
    trained, untrained = trained.raw.raw, untrained.raw.raw
    trained, untrained = trained.groupby('subject').median(), untrained.groupby('subject').median()
    # test difference
    t, p = ttest_ind(trained, untrained)
    logger.info(f"{model} difference on {benchmark}: t={t}, p={p}")
