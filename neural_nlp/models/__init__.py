import logging
import sys

import argparse
from brainio_base.assemblies import merge_data_arrays

from neural_nlp.models.implementations import _model_mappings, load_model, model_layers

_logger = logging.getLogger(__name__)


def get_activations(model_identifier, layers, stimuli):
    _logger.debug(f'Loading model {model_identifier}')
    model = load_model(model_identifier)

    _logger.debug("Retrieving activations")
    activations = model(stimuli, layers=layers)
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
    parser.add_argument('--model', type=str, required=True, choices=list(_model_mappings.keys()))
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
