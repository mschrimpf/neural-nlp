import argparse

import math
import random

import logging
import mkgu
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from neural_metrics.compare import pearsonr_matrix

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='IT')
    parser.add_argument('--variance', type=str, default='V6')
    parser.add_argument('--by_category', action='store_true', default=False,
                        help='Calculate correlation across each category independently')
    parser.add_argument('--no-by_category', action='store_false', dest='by_category',
                        help='Calculate correlation across all images')
    parser.add_argument('--cross_correlations', type=int, default=10)
    parser.add_argument('--regression_components', type=int, default=25)
    parser.add_argument('--test_size', type=int, default=0.25)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info("Running with args %s", vars(args))

    hvm = mkgu.get_assembly(name="HvM")
    hvm = hvm.sel(region=args.region, var=args.variance)
    hvm.load()

    if args.by_category:
        categories = np.unique(hvm.category)
        category_correlations = {}
        for category_iter, category in enumerate(categories):
            logger.debug("Category {}/{}: {}".format(category_iter + 1, args.cross_correlations, category))
            category_data = hvm.sel(category=category).squeeze('time_bin')
            correlations = []
            for split in range(args.cross_correlations):
                logger.debug("Split {}/{}".format(split + 1, args.cross_correlations))
                correlations.append(compare(category_data,
                                            regression_components=args.regression_components, test_size=args.test_size))
            category_correlations[category] = np.mean(correlations)
        logger.info("Category correlations: {}".format(category_correlations))
    else:
        category_data = hvm.squeeze('time_bin')
        correlations = []
        for split in range(args.cross_correlations):
            logger.debug("Split {}/{}".format(split + 1, args.cross_correlations))
            correlations.append(compare(category_data,
                                        regression_components=args.regression_components, test_size=args.test_size))
        logger.info("Correlations: {}".format(np.mean(correlations)))


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


if __name__ == '__main__':
    main()
