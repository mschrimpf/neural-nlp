import argparse
import logging
import sys

from result_caching import store_xarray

from brainscore.assemblies import merge_data_arrays
from neural_nlp.models.implementations import _model_mappings, load_model, model_layers
from neural_nlp.stimuli import _mappings, load_stimuli

_logger = logging.getLogger(__name__)


@store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer', 'sentences': 'sentence'})
def get_activations(model_name, layers, stimulus_set_name):
    _logger.debug('Loading model')
    model = load_model(model_name)
    _logger.debug('Loading stimuli')
    stimuli = load_stimuli(stimulus_set_name)

    _logger.debug("Retrieving activations")
    activations = []
    for i, sentence in enumerate(stimuli, start=1):
        if (i - 1) % 10 == 0:
            _logger.debug("Sentence {}/{} ({:.0f}%)".format(i, len(stimuli), 100 * i / len(stimuli)))
        sentence_activations = model.get_activations([sentence], layers=layers)
        assert len(sentence_activations) == 1
        activations.append(sentence_activations)

    _logger.debug("Merging assemblies")
    activations = merge_data_arrays(activations)
    return activations


def main():
    parser = argparse.ArgumentParser('model activations')
    parser.add_argument('--model', type=str, required=True, choices=list(_model_mappings.keys()))
    parser.add_argument('--layers', type=str, nargs='+', default='default')
    parser.add_argument('--dataset', type=str, required=True, choices=list(_mappings.keys()))
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    if args.layers == 'default':
        args.layers = model_layers[args.model]
    _logger.info("Running with args %s", vars(args))

    activations = get_activations(args.model, layers=args.layers, stimulus_set_name=args.dataset)
    _logger.info("Activations computed: {}".format(activations))


if __name__ == '__main__':
    main()
