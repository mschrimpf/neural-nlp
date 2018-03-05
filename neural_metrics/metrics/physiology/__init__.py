import argparse
import logging
import os
import sys
from collections import defaultdict

import mkgu
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from xarray import Dataset, DataArray

from neural_metrics import utils
from neural_metrics.metrics.similarities import pearsonr_matrix

_logger = logging.getLogger(__name__)


class _Defaults(object):
    regions = ['V4', 'IT']
    variance = 'V6'


class SimilarityWorker(object):
    def __init__(self, model_activations, basepath, regions, variance=_Defaults.variance,
                 output_directory=None, use_cached=True):
        self._variance = variance
        self._region_data = {region: _load_data(region=region)[1] for region in regions}
        raw_reference_data = _load_data(region=next(iter(self._region_data.keys())))[0]
        self._model_activations = model_activations
        self._model_activations = _rearrange_image_to_layer_object_image_activations(
            self._model_activations, raw_reference_data)
        self._cache = self.StorageCache(basepath, output_directory=output_directory, use_cached=use_cached)
        self._logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def __call__(self, layers, region, return_raw=False):
        if isinstance(layers, str):  # single layer passed
            layers = [layers]
        layers = tuple(layers)
        unmatched_layers = [layer for layer in layers if layer not in self._model_activations]
        if len(unmatched_layers) > 0:
            raise ValueError("Layers {} not found in activations".format(",".join(unmatched_layers)))
        if region not in self._region_data.keys():
            raise ValueError("Unknown region {}".format(region))

        cache_key = (layers, region, self._variance)
        if cache_key in self._cache:
            self._logger.debug("Similarity between {} and {} from cache".format(",".join(layers), region))
            _similarities = self._cache[cache_key]
        else:
            self._logger.debug("Computing similarity between {} and {}".format(",".join(layers), region))
            layers_activations = _merge_layer_activations(layers, self._model_activations)
            _similarities = _predictions_similarity(layers_activations, self._region_data[region])[1]
            self._cache[cache_key] = _similarities
        return _layer_correlation_meanstd(_similarities)[0] if not return_raw else _similarities

    def get_model_layers(self):
        return list(self._model_activations.keys())

    def get_savepath(self):
        return self._cache._savepath

    class StorageCache(utils.StorageCache):
        def __init__(self, basepath, output_directory=None, use_cached=True):
            savepath = get_savepath(basepath, output_directory=output_directory)
            super(SimilarityWorker.StorageCache, self).__init__(savepath=savepath, use_cached=use_cached)

        def __setitem__(self, key, value):
            assert len(key) == 3  # layers, region, variance
            super(SimilarityWorker.StorageCache, self).__setitem__(key, value)


def _predictions_similarity(object_activations, standardized_data):
    layer_activations = []
    neural_responses = []
    objects = []
    for obj, image_activations in object_activations.items():
        for image_id, image_activation in image_activations.items():
            layer_activations.append(image_activation)

            # spike count, averaged over multiple presentations
            neural_image_responses = standardized_data.sel(id=image_id)
            neural_responses.append(neural_image_responses)

            objects.append(obj)
    # fit all neuroids jointly
    layer_activations = np.array(layer_activations)
    neural_responses = np.array([n.data for n in neural_responses])
    cross_predictions = _split_predict(layer_activations, neural_responses, objects)
    similarities = _correlate(cross_predictions)
    return cross_predictions, similarities


def metrics_for_activations(activations_filepath, regions=_Defaults.regions, variance=_Defaults.variance,
                            output_directory=None, use_cached=True):
    similarities = SimilarityWorker(activations_filepath, regions=regions, variance=variance,
                                    output_directory=output_directory, use_cached=use_cached)
    for region in regions:
        _logger.debug('Region {}'.format(region))
        for layer in similarities.get_model_layers():
            _logger.debug('Layer {}'.format(layer))
            _similarities = similarities(layers=[layer], region=region, return_raw=True)
            mean, std = _layer_correlation_meanstd(_similarities)
            _logger.info("{} -> {}+-{}".format(layer, mean, std))
    return similarities.get_savepath()


def get_savepath(basepath, output_directory=None):
    save_name = os.path.splitext(os.path.basename(basepath))[0]
    output_directory = output_directory or os.path.dirname(basepath)
    savepath = os.path.join(output_directory, save_name + '-correlations.pkl')
    return savepath


def _merge_layer_activations(layers, layer_object_activations):
    # this is horribly ugly. Really need to restructure the data with pandas/xarrayF
    activations = {}
    for obj, image_activations in layer_object_activations[layers[0]].items():
        activations[obj] = {}
        for image_id in image_activations.keys():
            image_activations = [layer_object_activations[layer][obj][image_id] for layer in layers]
            activations[obj][image_id] = np.concatenate(image_activations) if len(image_activations) > 1 \
                else image_activations[0]
    return activations


def _rearrange_image_to_layer_object_image_activations(image_activations, hvm):
    layer_object_activations = defaultdict(lambda: defaultdict(dict))
    missing_hvm_image_paths = []
    for image_path, layer_activations in image_activations.items():
        image_id = _get_id_from_image_path(image_path)
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
        _logger.warning("{} images not found in neural recordings: {}"
                        .format(len(missing_hvm_image_paths), missing_paths))
    return layer_object_activations


def _get_id_from_image_path(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def _split_predict(source_responses, target_responses, object_labels, num_splits=10, max_components=200, test_size=.25):
    if source_responses.shape[1] > max_components:
        _logger.debug('PCA from {} to {}'.format(source_responses.shape[1], max_components))
        source_responses = PCA(n_components=max_components).fit_transform(source_responses)
    cross_validation = StratifiedShuffleSplit(n_splits=num_splits, test_size=test_size)
    results = []
    for split_iterator, (train_idx, test_idx) in enumerate(cross_validation.split(source_responses, object_labels)):
        _logger.debug('Fitting split {}/{}'.format(split_iterator, num_splits))
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


def _correlate(fitted_responses):
    correlations = []
    for split_data in fitted_responses:
        split = np.unique(split_data.split.data)
        assert len(split) == 1
        split = split[0]
        _logger.debug('Correlating split {}/{}'.format(split + 1, len(fitted_responses)))
        rs = pearsonr_matrix(split_data.target.data, split_data.prediction.data)
        correlations.append(DataArray(rs, dims=['neuroid'], coords={'neuroid': split_data.neuroid},
                                      attrs={'index': split_data.index, 'split': split}))
    return correlations


def _layer_correlation_meanstd(correlations):
    neuroid_medians = [np.median(correlation.data) for correlation in correlations]
    return np.mean(neuroid_medians), np.std(neuroid_medians)


def _layers_correlation_meanstd(layers_correlations):
    means, stds = [], []
    for layer_correlation in layers_correlations.values():
        mean, std = _layer_correlation_meanstd(layer_correlation)
        means.append(mean)
        stds.append(std)
    return means, stds


_data = None

_data_params = None, None


def _load_data(region, variance='V6'):
    global _data_params
    if _data is not None and _data_params == (region, variance):
        return _data

    raw_data = mkgu.get_assembly(name="HvM")
    raw_data = raw_data.sel(region=region).sel(var=variance)
    raw_data.load()
    standardized_data = raw_data.groupby('id').mean(dim='presentation').squeeze("time_bin")
    _data_params = (region, variance)
    return raw_data, standardized_data


def run(activations_filepaths, regions, variance=_Defaults.variance,
        output_directory=None, save_plot=False):
    from neural_metrics import plot_layer_correlations, results_dir
    if isinstance(activations_filepaths, str):  # single filepath
        activations_filepaths = [activations_filepaths]
    for activations_filepath in activations_filepaths:
        _logger.info("Processing {}".format(activations_filepath))
        try:
            savepath = metrics_for_activations(activations_filepath, regions=regions, variance=variance,
                                               output_directory=output_directory)
            file_name = os.path.splitext(os.path.basename(activations_filepath))[0]
            output_filepath = os.path.join(results_dir,
                                           '{}-physiology-regions_{}.{}'.format(file_name, ''.join(regions), 'svg'))
            plot_layer_correlations(savepath, output_filepath=output_filepath if save_plot else None)
        except Exception:
            _logger.exception("Error during {}, regions {}".format(activations_filepath, regions))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--activations_filepath', type=str, nargs='+',
                        default=[os.path.join(os.path.dirname(__file__), '..', '..', '..', 'images', 'sorted', 'Chairs',
                                              'vgg16-activations.pkl')],
                        help='one or more filepaths to the model activations')
    parser.add_argument('--output_directory', type=str, default=None,
                        help='directory to save results to. directory of activations_filepath if None')
    parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'], help='region(s) in brain to compare to')
    parser.add_argument('--variance', type=str, default=_Defaults.variance, help='type of images to compare to')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    _logger.info("Running with args %s", vars(args))
    run(activations_filepaths=args.activations_filepath, regions=args.regions, variance=args.variance, save_plot=True)


if __name__ == '__main__':
    main()
