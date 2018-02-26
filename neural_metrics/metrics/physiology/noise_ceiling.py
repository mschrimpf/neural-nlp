import argparse
import logging
import math
import random
import sys

import mkgu
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

from neural_metrics.metrics.similarities import pearsonr_matrix

_logger = logging.getLogger(__name__)


class _Defaults(object):
    region = 'IT'
    variance = 'V6'
    test_size = 0.25
    regression_components = 25
    cross_correlations = 10


def noise_ceiling_by_category(hvm=None, test_size=_Defaults.test_size,
                              regression_components=_Defaults.regression_components,
                              cross_correlations=_Defaults.cross_correlations):
    hvm = hvm if hvm is not None else _load_hvm()
    categories = np.unique(hvm.category)
    category_correlations = {}
    for category_iter, category in enumerate(categories):
        _logger.debug("Category {}/{}: {}".format(category_iter + 1, cross_correlations, category))
        category_data = hvm.sel(category=category).squeeze('time_bin')
        correlations = []
        for split in range(cross_correlations):
            _logger.debug("Split {}/{}".format(split + 1, cross_correlations))
            correlations.append(compare(category_data,
                                        regression_components=regression_components, test_size=test_size))
        category_correlations[category] = np.mean(correlations)
    return category_correlations


def noise_ceiling(hvm=None, test_size=_Defaults.test_size, regression_components=_Defaults.regression_components,
                  cross_correlations=_Defaults.cross_correlations):
    hvm = hvm if hvm is not None else _load_hvm()
    category_data = hvm.squeeze('time_bin')
    correlations = []
    for split in range(cross_correlations):
        _logger.debug("Split {}/{}".format(split + 1, cross_correlations))
        correlations.append(compare(category_data,
                                    regression_components=regression_components, test_size=test_size))
    return np.mean(correlations)


def compare(category_data, regression_components=25, test_size=0.25):
    nb_images = np.unique(category_data.id).shape[0]
    data1, data2 = [], []

    for id in np.unique(category_data.id.data):
        image_data = category_data.sel(id=id)
        presentations = set(image_data.presentation.data)
        presentations_half1 = set(random.sample(presentations, math.floor(len(presentations) / 2)))
        presentations_half2 = presentations - presentations_half1
        if len(presentations_half2) > len(presentations_half1):
            presentations_half2 = list(presentations_half2)[:-1]
        d1 = image_data.sel(presentation=list(presentations_half1))
        d2 = image_data.sel(presentation=list(presentations_half2))
        data1.append(d1.data.T.mean(axis=0))
        data2.append(d2.data.T.mean(axis=0))

    data1 = np.concatenate(data1).reshape(nb_images, -1)
    data2 = np.concatenate(data2).reshape(nb_images, -1)

    X_train, X_test, Y_train, Y_test = train_test_split(data1, data2, test_size=test_size)
    reg = PLSRegression(n_components=regression_components, scale=False)
    reg.fit(X_train, Y_train)
    pred = reg.predict(X_test)
    rs = pearsonr_matrix(Y_test, pred)

    return np.median(rs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default=_Defaults.region)
    parser.add_argument('--variance', type=str, default=_Defaults.variance)
    parser.add_argument('--by_category', action='store_true', default=False,
                        help='Calculate correlation across each category independently')
    parser.add_argument('--no-by_category', action='store_false', dest='by_category',
                        help='Calculate correlation across all images')
    parser.add_argument('--cross_correlations', type=int, default=_Defaults.cross_correlations)
    parser.add_argument('--regression_components', type=int, default=_Defaults.regression_components)
    parser.add_argument('--test_size', type=int, default=_Defaults.test_size)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running with args %s", vars(args))

    hvm = _load_hvm(region=args.region, variance=args.variance)

    if args.by_category:
        category_correlations = noise_ceiling_by_category(hvm, test_size=args.test_size,
                                                          regression_components=args.regression_components,
                                                          cross_correlations=args.cross_correlations)
        _logger.info("Category correlations: {}".format(category_correlations))
    else:
        ceiling = noise_ceiling(hvm, test_size=args.test_size, regression_components=args.regression_components,
                                cross_correlations=args.cross_correlations)
        _logger.info("Correlations: {}".format(ceiling))


def _load_hvm(region=_Defaults.region, variance=_Defaults.variance):
    hvm = mkgu.get_assembly(name="HvM")
    hvm = hvm.sel(region=region, var=variance)
    hvm.load()
    return hvm


if __name__ == '__main__':
    main()
