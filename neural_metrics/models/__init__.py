import argparse
import logging
import os
import re
import sys

from neural_metrics.models.implementations import model_mappings, prepare_images
from neural_metrics.models.outputs import get_model_outputs
from neural_metrics.models.type import ModelType, get_model_type
from neural_metrics.models.type import ModelType, get_model_type, PYTORCH_SUBMODULE_SEPARATOR
from neural_metrics.utils import save

_logger = logging.getLogger(__name__)


class _Defaults(object):
    model_weights = 'imagenet'
    pca_components = 200
    image_size = 224
    images_directory = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sorted')
    batch_size = 64


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

    activations_for_model(model=args.model, layers=args.layers, pca_components=args.pca,
                          image_size=args.image_size, images_directory=args.images_directory,
                          batch_size=args.batch_size)


def activations_for_model(model, layers, use_cached=False,
                          model_weights=_Defaults.model_weights, pca_components=_Defaults.pca_components,
                          image_size=_Defaults.image_size, images_directory=_Defaults.images_directory,
                          batch_size=_Defaults.batch_size):
    args = locals()
    savepath = get_savepath(model, model_weights, images_directory)
    if use_cached and os.path.isfile(savepath):
        _logger.info('Using cached activations: {}'.format(savepath))
        return savepath
    # model
    _logger.debug('Creating model')
    model, preprocess_input = model_mappings[model](image_size, weights=model_weights)
    print_verify_model(model, layers)
    model_type = get_model_type(model)
    # input
    _logger.debug('Loading input images')
    image_filepaths, images = prepare_images(images_directory, image_size, preprocess_input, model_type)
    # output
    _logger.debug('Computing activations')
    layer_outputs = get_model_outputs(model, images, layers, batch_size=batch_size, pca_components=pca_components)
    stimuli_layer_activations = {}
    for i, image_filepath in enumerate(image_filepaths):
        image_relpath = os.path.relpath(image_filepath, images_directory)
        stimuli_layer_activations[image_relpath] = {layer_name: layer_outputs[i]
                                                    for layer_name, layer_outputs in layer_outputs.items()}
    save({'activations': stimuli_layer_activations, 'args': args}, savepath)
    return savepath


def print_verify_model(model, layer_names):
    model_type = get_model_type(model)
    _print_verify_model = {ModelType.KERAS: print_verify_model_keras,
                           ModelType.PYTORCH: print_verify_model_pytorch}[model_type]
    _print_verify_model(model, layer_names)


def print_verify_model_pytorch(model, layer_names):
    _logger.debug(str(model))

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


def print_verify_model_keras(model, layer_names):
    model.summary(print_fn=_logger.debug)
    nonexisting_layers = set(layer_names) - set([layer.name for layer in model.layers])
    assert len(nonexisting_layers) == 0, "Layers not found in keras model: %s" % str(nonexisting_layers)


def get_savepath(model, model_weights=_Defaults.model_weights, images_directory=_Defaults.images_directory):
    return os.path.join(images_directory, '{}-weights_{}-activations.pkl'.format(model, model_weights))


def model_name_from_activations_filepath(activations_filepath):
    match = re.match('^(.*)-weights_[^-]*-activations.pkl$', os.path.basename(activations_filepath))
    if not match:
        raise ValueError("Filename {} did not match".format(os.path.basename(activations_filepath)))
    return match.group(1)


if __name__ == '__main__':
    main()
