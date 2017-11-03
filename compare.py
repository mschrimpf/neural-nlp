import argparse
import os
import pickle
import warnings
from collections import defaultdict

import mkgu
import numpy as np
import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

from utils import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_filepath', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'images', 'sorted', 'Chairs',
                                             'vgg16-activations.pkl'))
    parser.add_argument('--region', type=str, default='IT')
    parser.add_argument('--variance', type=str, default='V6')
    args = parser.parse_args()
    print("Running with args", args)

    # neural data
    hvm = mkgu.get_assembly(name="HvM")
    hvm = hvm.sel(region=args.region).sel(var=args.variance)
    hvm.load()
    hvm = hvm.groupby('id').mean(dim='presentation').squeeze("time_bin")

    # model data
    image_activations = load_image_activations(args.activations_filepath)
    layer_object_activations = defaultdict(lambda: defaultdict(dict))
    for image_path, layer_activations in image_activations.items():
        image_id = get_id_from_image_path(image_path)
        if image_id not in hvm.id.values:
            warnings.warn("Image %s not found in neural recordings" % image_path)
            continue
        obj = np.unique(hvm.sel(id=image_id).obj.data)
        assert len(obj) == 1
        for layer, image_activations in layer_activations.items():
            layer_object_activations[layer][obj[0]][image_id] = image_activations.flatten()

    # compare
    layer_metrics = {}
    for layer, object_activations in layer_object_activations.items():
        print('Layer %s' % layer)
        layer_activations = []
        neural_responses = []
        objects = []
        for obj, image_activations in object_activations.items():
            for image_id, image_activation in image_activations.items():
                layer_activations.append(image_activation)

                neural_image_responses = hvm.sel(id=image_id)
                neural_responses.append(neural_image_responses)  # spike count, averaged over multiple presentations

                objects.append(obj)
        # fit all neuroids separately
        layer_neuroid_metrics = {}
        for neuroid in neural_responses[0].neuroid:
            neuroid_responses = [object_responses.sel(neuroid=neuroid).data for object_responses in neural_responses]
            layer_neuroid_metrics[neuroid] = compare(np.array(layer_activations), np.array(neuroid_responses), objects)
        layer_metrics[layer] = np.median(list(layer_neuroid_metrics.values()))
        print("%s -> %f" % (layer, layer_metrics[layer]))
    save(layer_metrics, args.activations_filepath.replace('.pkl', '-correlations.pkl'))


def compare(layer_activations, neural_responses, object_labels, splits=10, max_components=200, test_size=.25):
    if layer_activations.shape[1] > max_components:
        layer_activations = PCA(n_components=max_components).fit_transform(layer_activations)
    cross_validation = StratifiedShuffleSplit(n_splits=splits, test_size=test_size)
    correlations = []
    for it, (train_idx, test_idx) in enumerate(cross_validation.split(layer_activations, object_labels)):
        reg = PLSRegression(n_components=25, scale=False)
        reg.fit(layer_activations[train_idx], neural_responses[train_idx])
        pred = reg.predict(layer_activations[test_idx])
        rs = pearsonr_matrix(np.expand_dims(neural_responses[test_idx], 1), pred)
        correlations.append(rs)
    return np.mean(correlations)  # TODO: mean here?


def pearsonr_matrix(data1, data2, axis=1):
    rs = []
    for i in range(data1.shape[axis]):
        d1 = np.take(data1, i, axis=axis)
        d2 = np.take(data2, i, axis=axis)
        r, p = scipy.stats.pearsonr(d1, d2)
        rs.append(r)
    return np.array(rs)


def load_image_activations(activations_filepath):
    with open(activations_filepath, 'rb') as file:
        image_activations = pickle.load(file)
    return image_activations


def get_id_from_image_path(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


if __name__ == '__main__':
    main()
