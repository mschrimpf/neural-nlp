import argparse
import logging
import os
import pickle
import re
import sys
from collections import defaultdict

from neural_metrics.models.implementations import model_mappings, prepare_images
from neural_metrics.models.outputs import get_model_outputs
from neural_metrics.models.type import ModelType, get_model_type
from neural_metrics.models.type import ModelType, get_model_type, PYTORCH_SUBMODULE_SEPARATOR
from neural_metrics.utils import StorageCache

_logger = logging.getLogger(__name__)


class _Defaults(object):
    model_weights = 'imagenet'
    pca_components = 200
    image_size = 224
    images_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sorted')
    batch_size = 64


class ActivationsWorker(object):
    def __init__(self, model_name, model_weights=_Defaults.model_weights,
                 image_size=_Defaults.image_size, images_directory=_Defaults.images_directory):
        self._logger = logging.getLogger(__class__.__name__)
        self._model_name = model_name
        self._model_weights = model_weights
        self._images_directory = images_directory
        # model
        self._model, preprocess_input = model_mappings[model_name](image_size, weights=model_weights)
        _print_model(self._model)
        model_type = get_model_type(self._model)
        # images
        self._logger.debug('Loading input images')
        self._image_filepaths, self._images = prepare_images(images_directory, image_size, preprocess_input, model_type)
        # cache
        self._savepath = get_savepath(model_name, model_weights, images_directory)
        self._cache = StorageCache(self._savepath)

    def __call__(self, layers, pca_components=_Defaults.pca_components, batch_size=_Defaults.batch_size):
        _verify_model_layers(self._model, layers)
        # compute missing
        uncomputed_layers = layers if len(self._cache) == 0 else \
            set(layers) - set((layer for layer, imagepath in self._cache.keys()))
        if len(uncomputed_layers) > 0:
            self._logger.debug('Computing activations for layers {}'.format(",".join(uncomputed_layers)))
            activations = get_model_outputs(self._model, self._images, uncomputed_layers,
                                            batch_size=batch_size, pca_components=pca_components)
            for i, image_filepath in enumerate(self._image_filepaths):
                image_relpath = os.path.relpath(image_filepath, self._images_directory)
                for layer_name, layer_outputs in activations.items():
                    self._cache[(layer_name, image_relpath)] = layer_outputs[i]

        # re-arrange
        stimuli_layer_activations = defaultdict(dict)
        for image_filepath in self._image_filepaths:
            image_relpath = os.path.relpath(image_filepath, self._images_directory)
            for layer_name in layers:
                stimuli_layer_activations[image_relpath][layer_name] = self._cache[(layer_name, image_relpath)]
        return stimuli_layer_activations

    def get_savepath(self):
        return self._savepath


def activations_for_model(model, layers, model_weights=_Defaults.model_weights,
                          image_size=_Defaults.image_size, images_directory=_Defaults.images_directory,
                          pca_components=_Defaults.pca_components, batch_size=_Defaults.batch_size):
    worker = ActivationsWorker(model_name=model, model_weights=model_weights,
                               image_size=image_size, images_directory=images_directory)
    return worker(layers=layers, pca_components=pca_components, batch_size=batch_size)


def _print_model(model):
    model_type = get_model_type(model)
    if model_type == ModelType.KERAS:
        model.summary(print_fn=_logger.debug)
    elif model_type == ModelType.PYTORCH:
        _logger.debug(str(model))
    else:
        raise ValueError()


def _verify_model_layers(model, layer_names):
    model_type = get_model_type(model)
    _verify_model = {ModelType.KERAS: _verify_model_layers_keras,
                     ModelType.PYTORCH: _verify_model_layers_pytorch}[model_type]
    _verify_model(model, layer_names)


def _verify_model_layers_pytorch(model, layer_names):
    def collect_pytorch_layer_names(module, parent_module_parts):
        result = []
        for submodule_name, submodule in module._modules.items():
            if not hasattr(submodule, '_modules') or len(submodule._modules) == 0:
                result.append(PYTORCH_SUBMODULE_SEPARATOR.join(parent_module_parts + [submodule_name]))
            else:
                result += collect_pytorch_layer_names(submodule, parent_module_parts + [submodule_name])
        return result

    nonexisting_layers = set(layer_names) - set(collect_pytorch_layer_names(model, []))
    assert len(nonexisting_layers) == 0, "Layers not found in PyTorch model: %s" % str(nonexisting_layers)


def _verify_model_layers_keras(model, layer_names):
    nonexisting_layers = set(layer_names) - set([layer.name for layer in model.layers])
    assert len(nonexisting_layers) == 0, "Layers not found in keras model: %s" % str(nonexisting_layers)


def get_savepath(model, model_weights=_Defaults.model_weights, images_directory=_Defaults.images_directory):
    return os.path.join(images_directory, '{}-weights_{}-activations.pkl'.format(model, model_weights))


def model_name_from_activations_filepath(activations_filepath):
    match = re.match('^(.*)-weights_[^-]*-activations.pkl$', os.path.basename(activations_filepath))
    if not match:
        raise ValueError("Filename {} did not match".format(os.path.basename(activations_filepath)))
    return match.group(1)


def load_model_activations(activations_filepath):
    with open(activations_filepath, 'rb') as file:
        image_activations = pickle.load(file)
    return image_activations['activations']


def main():
    parser = argparse.ArgumentParser('model comparison')
    parser.add_argument('--model', type=str, required=True, choices=list(model_mappings.keys()))
    parser.add_argument('--model_weights', type=str, default=_Defaults.model_weights)
    parser.add_argument('--no-model_weights', action='store_const', const=None, dest='model_weights')
    parser.add_argument('--layers', nargs='+', required=True)
    parser.add_argument('--pca', type=int, default=_Defaults.pca_components,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--image_size', type=int, default=_Defaults.image_size)
    parser.add_argument('--images_directory', type=str, default=_Defaults.images_directory)
    parser.add_argument('--batch_size', type=int, default=_Defaults.batch_size)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    _logger.info("Running with args %s", vars(args))

    activations_for_model(model=args.model, layers=args.layers, image_size=args.image_size,
                          images_directory=args.images_directory, pca_components=args.pca, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
