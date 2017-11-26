import argparse

import math
import random
import mkgu
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from compare import pearsonr_matrix

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='IT')
    parser.add_argument('--variance', type=str, default='V6')
    parser.add_argument('--by_category', type=str, choices=['True', 'False'], default='False',
                        help='Calculate correlation across each category independently (True), '
                             'or across all images (False)')
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    print("Running with args %s", vars(args))

    hvm = mkgu.get_assembly(name="HvM")
    hvm = hvm.sel(region=args.region, var=args.variance)
    hvm.load()

    random_split_trials = args.trials
    category_filters = [
    'Animals',
    'Boats',
    'Cars',
    'Chairs',
    'Faces',
    'Fruits',
    'Planes',
    'Tables'
    ]

    category_correlations = {}

    if args.by_category == 'True':

        for category in category_filters:
            category_data = hvm.sel(category=category).squeeze('time_bin')
            correlations = []

            for i in range(random_split_trials):
                correlations.append(compare(category_data))
            category_correlations[category] = np.mean(correlations)

        print(category_correlations)

    else:
        category_data = hvm.squeeze('time_bin')
        correlations = []

        for i in range(random_split_trials):
            correlations.append(compare(category_data))
        print(np.mean(correlations))

def compare(category_data):
        nb_images = np.unique(category_data.id).shape[0]
        nb_neuroids = 168
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

        data1 = np.concatenate(data1).reshape(nb_images, nb_neuroids)
        data2 = np.concatenate(data2).reshape(nb_images, nb_neuroids)

        X_train, X_test, Y_train, Y_test = train_test_split(data1, data2, test_size=0.3)
        reg = PLSRegression(n_components=25, scale=False)
        reg.fit(X_train, Y_train)
        pred = reg.predict(X_test)
        rs = pearsonr_matrix(Y_test, pred)

        return np.median(rs)

if __name__ == '__main__':
    main()