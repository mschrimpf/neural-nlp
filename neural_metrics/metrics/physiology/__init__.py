import argparse
import logging
import os
import pickle
import sys
from collections import OrderedDict, defaultdict

import mkgu
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from xarray import Dataset, DataArray

from neural_metrics.metrics.similarities import pearsonr_matrix
from neural_metrics.utils import save

_logger = logging.getLogger(__name__)


class _Defaults(object):
    region = 'IT'
    variance = 'V6'
    concat_up_to_n_layers = 1


class SimilarityWorker(object):
    def __init__(self, activations_filepath, regions, variance=_Defaults.variance, use_cached=True):
        self._region_data = {region: _load_data(region=region)[1] for region in regions}
        raw_reference_data = _load_data(region=next(iter(self._region_data.keys())))[0]
        self._model_activations = load_model_activations(activations_filepath)
        self._model_activations = rearrange_image_to_layer_object_image_activations(
            self._model_activations, raw_reference_data)
        self._cache = self.StorageCache(activations_filepath, regions, variance, use_cached=use_cached)
        self._logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def __call__(self, layers, region):
        if isinstance(layers, str):  # single layer passed
            layers = [layers]
        layers = tuple(layers)
        unmatched_layers = [layer for layer in layers if layer not in self._model_activations]
        if len(unmatched_layers) > 0:
            raise ValueError("Layers {} not found in activations".format(",".join(unmatched_layers)))
        if region not in self._region_data.keys():
            raise ValueError("Unknown region {}".format(region))

        if (layers, region) in self._cache:
            self._logger.debug("Similarity between {} and {} from cache".format(",".join(layers), region))
            _similarities = self._cache[(layers, region)]
        else:
            self._logger.debug("Computing similarity between {} and {}".format(",".join(layers), region))
            layers_activations = _merge_layer_activations(layers, self._model_activations)
            _similarities = predictions_similarity(layers_activations, self._region_data[region])[1]
            self._cache[(layers, region)] = _similarities
        return layer_correlation_meanstd(_similarities)[0]

    def get_model_layers(self):
        return list(self._model_activations.keys())

    class StorageCache(dict):
        """
        Keeps computed similarities in memory (the cache) and writes them to a file (the storage).
        """

        def __init__(self, activations_filepath, regions, variance, use_cached=True):
            super(SimilarityWorker.StorageCache, self).__init__()
            self._activations_filepath, self._variance = activations_filepath, variance
            self._logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
            self._setitem_from_storage = False  # marks when items are set from loading, i.e. we do not have to save
            if use_cached:
                for region in regions:
                    self._load_storage(region)

        def __setitem__(self, key, value):
            super(SimilarityWorker.StorageCache, self).__setitem__(key, value)
            if self._setitem_from_storage:
                return
            layers, region = key
            self._save_storage(region)

        def _load_storage(self, region):
            region_savepath = self._savepath(region)
            if not os.path.isfile(region_savepath):
                self._logger.debug("Region {} not stored in memory".format(region))
                return
            self._logger.debug("Loading region {} from storage: {}".format(region, region_savepath))
            with open(region_savepath, 'rb') as f:
                region_storage = pickle.load(f)
            for layers, stored_values in region_storage['data'].items():
                self._setitem_from_storage = True
                self[(layers, region)] = stored_values
                self._setitem_from_storage = False

        def _save_storage(self, region):
            savepath = self._savepath(region)
            self._logger.debug("Saving region {} to storage: {}".format(region, savepath))
            region_storage = {}
            for (layers, _region), values in self.items():
                if _region == region:
                    region_storage[layers] = values
            savepath_part = savepath + '.filepart'  # write first to avoid partial writes
            with open(savepath_part, 'wb') as f:
                pickle.dump({'data': region_storage, 'args': {'region': region, 'variance': self._variance}}, f)
            os.rename(savepath_part, savepath)

        def _savepath(self, region):
            return get_savepath(self._activations_filepath, region=region, variance=self._variance)


def predictions_similarity(object_activations, standardized_data):
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
    cross_predictions = split_predict(layer_activations, neural_responses, objects)
    similarities = correlate(cross_predictions)
    return cross_predictions, similarities


def layers_from_raw_activations(model_activations):
    # model_activations: image_path -> {layer -> activations}
    return list(model_activations[next(iter(model_activations.keys()))].keys())


def metrics_for_activations(activations_filepath, use_cached=True,
                            region=_Defaults.region, variance=_Defaults.variance,
                            output_directory=None, ignore_layers=None,
                            concat_up_to_n_layers=_Defaults.concat_up_to_n_layers):
    args = locals()
    savepath = get_savepath(activations_filepath, region, variance, output_directory)
    if use_cached and os.path.isfile(savepath):
        _logger.info('Using cached activations: {}'.format(savepath))
        return savepath
    # neural data
    raw_data, standardized_data = _load_data(region=region, variance=variance)
    # model data
    image_activations = load_model_activations(activations_filepath)
    layer_object_activations = rearrange_image_to_layer_object_image_activations(image_activations, raw_data)
    # prepare combined activations
    comparison_basis = prepare_layer_activations(layer_object_activations, concat_up_to_n_layers, ignore_layers)
    # compare
    layer_metrics, layer_predictions = OrderedDict(), OrderedDict()
    for layers, object_activations in comparison_basis.items():
        _logger.debug('Layer{} {}'.format('s' if len(layers) > 1 else '', layers if len(layers) > 1 else layers[0]))
        cross_predictions, similarities = predictions_similarity(object_activations, standardized_data)
        layer_predictions[layers] = cross_predictions
        layer_metrics[layers] = similarities
        mean, std = layer_correlation_meanstd(similarities)
        _logger.info("{} -> {}+-{}".format(layers, mean, std))
    _logger.debug('Saving to {}'.format(savepath))
    save({'args': args, 'layer_metrics': layer_metrics, 'layer_predictions': layer_predictions}, savepath)
    return savepath


def get_savepath(activations_filepath, region, variance, output_directory=None):
    [save_name, save_ext] = os.path.splitext(os.path.basename(activations_filepath))
    output_directory = output_directory or os.path.dirname(activations_filepath)
    savepath = os.path.join(output_directory, save_name + '-correlations-region_{}-variance_{}{}'.format(
        region, variance, save_ext))
    return savepath


def prepare_layer_activations(layer_object_activations, concat_up_to_n_layers, ignore_layers):
    comparison_layers = [layer for layer in layer_object_activations.keys()
                         if not ignore_layers or layer not in ignore_layers]
    if len(comparison_layers) != len(layer_object_activations):
        _logger.debug('Ignoring layer(s) {}'.format(", ".join(
            [layer for layer in layer_object_activations.keys() if layer not in comparison_layers])))
    comparison_basis = OrderedDict()
    for layer_num, layer in enumerate(comparison_layers):
        for max_layer in range(layer_num, min(layer_num + concat_up_to_n_layers, len(comparison_layers))):
            _logger.debug("Concatenating from layer {} ({}) up to {}".format(layer_num, layer, max_layer))
            layers = tuple(comparison_layers[layer_num:max_layer + 1])
            activations = _merge_layer_activations(layers, layer_object_activations)
            comparison_basis[layers] = activations
    return comparison_basis


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
        _logger.warning("{} images not found in neural recordings: {}"
                        .format(len(missing_hvm_image_paths), missing_paths))
    return layer_object_activations


def split_predict(source_responses, target_responses, object_labels, num_splits=10, max_components=200, test_size=.25):
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


def correlate(fitted_responses):
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


def load_model_activations(activations_filepath):
    with open(activations_filepath, 'rb') as file:
        image_activations = pickle.load(file)
    return image_activations['activations']


def get_id_from_image_path(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


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
        concat_up_to_n_layers=_Defaults.concat_up_to_n_layers,
        output_directory=None, ignore_layers=None,
        save_plot=False):
    from neural_metrics import plot_layer_correlations, results_dir
    if isinstance(activations_filepaths, str):  # single filepath
        activations_filepaths = [activations_filepaths]
    for activations_filepath in activations_filepaths:
        _logger.info("Processing {}".format(activations_filepath))
        savepaths = []
        for region in regions:
            try:
                savepath = metrics_for_activations(activations_filepath,
                                                   region=region, variance=variance,
                                                   concat_up_to_n_layers=concat_up_to_n_layers,
                                                   output_directory=output_directory,
                                                   ignore_layers=ignore_layers)
                savepaths.append(savepath)
            except Exception:
                _logger.exception("Error during {}, region {}".format(activations_filepath, region))
        file_name = os.path.splitext(os.path.basename(activations_filepath))[0]
        output_filepath = os.path.join(results_dir,
                                       '{}-physiology-regions_{}.{}'.format(file_name, ''.join(regions), 'svg'))
        plot_layer_correlations(savepaths, labels=regions, output_filepath=output_filepath if save_plot else None)


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
    parser.add_argument('--concat_up_to_n_layers', type=int, default=1)
    parser.add_argument('--ignore_layers', type=str, nargs='+', default=None)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    _logger.info("Running with args %s", vars(args))
    run(activations_filepaths=args.activations_filepath, regions=args.regions, variance=args.variance,
        ignore_layers=args.ignore_layers, save_plot=True)


if __name__ == '__main__':
    main()
