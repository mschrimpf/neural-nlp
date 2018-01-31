import argparse
import logging
import os
import pickle
import sys
from collections import defaultdict, OrderedDict

import mkgu
import numpy as np
import scipy.stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from xarray import Dataset, DataArray

from neural_metrics.utils import save

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--activations_filepath', type=str, nargs='+',
                        default=[os.path.join(os.path.dirname(__file__), '..', 'images', 'sorted', 'Chairs',
                                              'vgg16-activations.pkl')],
                        help='one or more filepaths to the model activations')
    parser.add_argument('--output_directory', type=str, default=None,
                        help='directory to save results to. directory of activations_filepath if None')
    parser.add_argument('--region', type=str, default='IT', help='region in brain to compare to')
    parser.add_argument('--variance', type=str, default='V6', help='type of images to compare to')
    parser.add_argument('--ignore_layers', type=str, nargs='+', default=[])
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logger.info("Running with args %s", vars(args))

    # neural data
    raw_data = mkgu.get_assembly(name="HvM")
    raw_data = raw_data.sel(region=args.region).sel(var=args.variance)
    raw_data.load()
    standardized_data = raw_data.groupby('id').mean(dim='presentation').squeeze("time_bin")

    for activations_filepath in args.activations_filepath:
        # model data
        image_activations = load_image_activations(activations_filepath)['activations']
        layer_object_activations = rearrange_image_to_layer_object_image_activations(image_activations, raw_data)

        # compare
        layer_metrics, layer_predictions = OrderedDict(), OrderedDict()
        for layer, object_activations in layer_object_activations.items():
            if layer in args.ignore_layers:
                logger.debug('Ignoring layer %s', layer)
                continue
            logger.debug('Layer %s' % layer)
            layer_activations = []
            neural_responses = []
            objects = []
            for obj, image_activations in object_activations.items():
                for image_id, image_activation in image_activations.items():
                    layer_activations.append(image_activation)

                    neural_image_responses = standardized_data.sel(id=image_id)
                    neural_responses.append(neural_image_responses)  # spike count, averaged over multiple presentations

                    objects.append(obj)
            # fit all neuroids jointly
            layer_activations = np.array(layer_activations)
            neural_responses = np.array([n.data for n in neural_responses])
            cross_predictions = split_predict(layer_activations, neural_responses, objects)
            layer_predictions[layer] = cross_predictions
            layer_metrics[layer] = correlate(cross_predictions)
            mean, std = layer_correlation_meanstd(layer_metrics[layer])
            logger.info("%s -> %f+-%f" % (layer, mean, std))

        [save_name, save_ext] = os.path.splitext(os.path.basename(activations_filepath))
        output_directory = args.output_directory or os.path.dirname(activations_filepath)
        savepath = os.path.join(output_directory, save_name + '-correlations-region_{}-variance_{}{}'.format(
            args.region, args.variance, save_ext))
        logger.debug('Saving to %s', savepath)
        save({'args': args, 'layer_metrics': layer_metrics, 'layer_predictions': layer_predictions}, savepath)


def rearrange_image_to_layer_object_image_activations(image_activations, hvm):
    layer_object_activations = defaultdict(lambda: defaultdict(dict))
    missing_hvm_image_paths = []
    for image_path, layer_activations in image_activations.items():
        image_id = get_id_from_image_path(image_path)
        if image_id not in hvm.id.values:
            missing_hvm_image_paths.append(image_path)
            continue
        obj = np.unique(hvm.sel(id=image_id).obj.data)
        assert len(obj) == 1
        for layer, image_activations in layer_activations.items():
            layer_object_activations[layer][obj[0]][image_id] = image_activations.flatten()
    if len(missing_hvm_image_paths) > 0:
        missing_paths = ", ".join(missing_hvm_image_paths)
        if len(missing_paths) > 300:
            missing_paths = missing_paths[:300] + "..."
        logger.warning("%d images not found in neural recordings: %s", len(missing_hvm_image_paths), missing_paths)
    return layer_object_activations


def split_predict(source_responses, target_responses, object_labels, num_splits=10, max_components=200, test_size=.25):
    if source_responses.shape[1] > max_components:
        source_responses = PCA(n_components=max_components).fit_transform(source_responses)
    cross_validation = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)
    results = []
    for split_iterator, (train_idx, test_idx) in enumerate(cross_validation.split(source_responses, object_labels)):
        logger.debug('Fitting split %d/%d', split_iterator, num_splits)
        reg = PLSRegression(n_components=25, scale=False)
        reg.fit(source_responses[train_idx], target_responses[train_idx])
        predicted_responses = reg.predict(source_responses[test_idx])
        results.append(Dataset({'source': (['index', 'source_dim'], source_responses[test_idx]),
                                'prediction': (['index', 'neuroid'], predicted_responses),
                                'target': (['index', 'neuroid'], target_responses[test_idx])},
                               coords={'index': test_idx, 'neuroid': range(target_responses.shape[1]),
                                       'source_dim': range(source_responses.shape[1]),
                                       'split': split_iterator}))
    return results


def correlate(fitted_responses):
    correlations = []
    for split_data in fitted_responses:
        split = np.unique(split_data.split.data)
        logger.debug('Correlating split %d/%d', split, len(fitted_responses))
        rs = pearsonr_matrix(split_data.target.data, split_data.prediction.data)
        correlations.append(DataArray(rs, dims=['neuroid'], coords={'neuroid': split_data.neuroid},
                                      attrs={'index': split_data.index, 'split': split}))
    return correlations


def layer_correlation_meanstd(correlations):
    neuroid_medians = [np.median(correlation.data) for correlation in correlations]
    return np.mean(neuroid_medians), np.std(neuroid_medians)


def layers_correlation_meanstd(layers_correlations):
    means, stds = [], []
    for layer_correlation in layers_correlations.values():
        mean, std = layer_correlation_meanstd(layer_correlation)
        means.append(mean)
        stds.append(std)
    return means, stds


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
