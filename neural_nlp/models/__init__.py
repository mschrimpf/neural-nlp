import logging
import sys
from collections import OrderedDict

import argparse
from brainio_base.assemblies import merge_data_arrays
from numpy.random.mtrand import RandomState

from neural_nlp.models.implementations import model_pool, load_model, model_layers

_logger = logging.getLogger(__name__)


class SubsamplingHook:
    def __init__(self, activations_extractor, num_features):
        self._activations_extractor = activations_extractor
        self._num_features = num_features
        self._sampling_indices = None

    def __call__(self, activations):
        self._ensure_initialized(activations)
        activations = OrderedDict((layer, layer_activations[:, self._sampling_indices[layer]])
                                  for layer, layer_activations in activations.items())
        return activations

    @classmethod
    def hook(cls, activations_extractor, num_features):
        hook = SubsamplingHook(activations_extractor=activations_extractor, num_features=num_features)
        handle = activations_extractor.register_activations_hook(hook)
        hook.handle = handle
        if hasattr(activations_extractor, 'identifier'):
            activations_extractor.identifier += f'-subsample_{num_features}'
        else:
            activations_extractor._extractor.identifier += f'-subsample_{num_features}'
        return handle

    def _ensure_initialized(self, activations):
        if self._sampling_indices:
            return
        rng = RandomState(0)
        self._sampling_indices = {layer: rng.randint(layer_activations.shape[1], size=self._num_features)
                                  for layer, layer_activations in activations.items()}


def get_activations(model_identifier, layers, stimuli, stimuli_identifier=None, subsample=None):
    _logger.debug(f'Loading model {model_identifier}')
    model = load_model(model_identifier)
    if subsample:
        SubsamplingHook.hook(model, subsample)

    _logger.debug("Retrieving activations")
    activations = model(stimuli, layers=layers, stimuli_identifier=stimuli_identifier)
    return activations


def get_activations_for_sentence(model_name, layers, sentences):
    model = load_model(model_name)
    activations = []
    for sentence in sentences:
        sentence_activations = model.get_activations([sentence], layers=layers)
        activations.append(sentence_activations)
    activations = merge_data_arrays(activations)
    return activations


def main():
    parser = argparse.ArgumentParser('model activations')
    parser.add_argument('--model', type=str, required=True, choices=list(model_pool.keys()))
    parser.add_argument('--layers', type=str, nargs='+', default='default')
    parser.add_argument('--sentence', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    if args.layers == 'default':
        args.layers = model_layers[args.model]
    _logger.info("Running with args %s", vars(args))

    activations = get_activations_for_sentence(args.model, layers=args.layers, sentences=args.sentence.split(' '))
    _logger.info("Activations computed: {}".format(activations))


if __name__ == '__main__':
    main()
