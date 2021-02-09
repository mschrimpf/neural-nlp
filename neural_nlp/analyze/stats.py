import itertools
import logging
import numpy as np
import pandas as pd
import scipy.stats
from numpy.random import RandomState
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

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
    from neural_nlp.analyze.scores.bars import retrieve_scores
    scores = retrieve_scores(benchmark, models=[model, f"{model}-untrained"])
    untrained_rows = np.array([model.endswith('-untrained') for model in scores['model']])
    trained, untrained = scores[~untrained_rows], scores[untrained_rows]
    logger.info(f"{model} trained score: {trained['score'].squeeze()}, untrained score: {untrained['score'].squeeze()}")


def interaction_test(data, category_column='category', compare_only=None, num_bootstraps=1000, bootstrap_size=.9):
    """
    :param data: pandas DataFrame with x and y values and the category column
    :param category_column:
    :param compare_only: instead of all pair-wise tests, compare all other categories only with this category
    :param num_bootstraps: number of bootstraps to determine regression parameters
    :param bootstrap_size: size of each bootstrap to determine regression parameters
    :return: a pandas DataFrame with compared categories and the corresponding test values
    """
    groups = dict(list(data.groupby(category_column)))
    result = []
    if compare_only:
        pairs = [dict((item, (compare_only, groups[compare_only])))
                 for item in groups.items() if item[0] != compare_only]
    else:
        pairs = list(map(dict, itertools.combinations(groups.items(), 2)))
    for pair in tqdm(pairs, desc=f'pair interactions'):
        descriptor1, descriptor2 = pair.keys()
        data1, data2 = pair.values()
        x1, y1 = data1['x'].values, data1['y'].values
        x2, y2 = data2['x'].values, data2['y'].values
        x1, x2 = scipy.stats.zscore(x1), scipy.stats.zscore(x2)  # normalize to control for different variances
        # run bootstraps to collect multiple samples of slope/intercept
        rng = RandomState(0)
        bootstraps = []
        for bootstrap in range(num_bootstraps):
            indices1 = rng.choice(np.arange(len(x1)), size=int(bootstrap_size * len(x1)))
            indices2 = rng.choice(np.arange(len(x2)), size=int(bootstrap_size * len(x2)))
            bootstrap_x1, bootstrap_y1 = x1[indices1], y1[indices1]
            bootstrap_x2, bootstrap_y2 = x2[indices2], y2[indices2]
            regression1 = LinearRegression().fit(np.expand_dims(bootstrap_x1, 1), bootstrap_y1)
            regression2 = LinearRegression().fit(np.expand_dims(bootstrap_x2, 1), bootstrap_y2)
            slope1, slope2 = regression1.coef_, regression2.coef_
            intercept1, intercept2 = regression1.intercept_, regression2.intercept_
            bootstraps.append({'bootstrap': bootstrap,
                               'slope1': slope1, 'slope2': slope2, 'intercept1': intercept1, 'intercept2': intercept2})
        bootstraps = pd.DataFrame(bootstraps)
        ttest_slope = ttest_ind(bootstraps['slope1'].values, bootstraps['slope2'].values)
        ttest_int = ttest_ind(bootstraps['intercept1'].values, bootstraps['intercept2'].values)
        result.append({'data1': descriptor1, 'data2': descriptor2, 'test': 'ttest_ind',
                       'slope1': np.mean(bootstraps['slope1']), 'slope2': np.mean(bootstraps['slope2']),
                       'intercept1': np.mean(bootstraps['intercept1']), 'intercept2': np.mean(bootstraps['intercept2']),
                       'p_slope': ttest_slope.pvalue.squeeze(), 'statistic_slope': ttest_slope.statistic.squeeze(),
                       'p_intercept': ttest_int.pvalue.squeeze(), 'statistic_intercept': ttest_int.statistic.squeeze()})
    return pd.DataFrame(result)
