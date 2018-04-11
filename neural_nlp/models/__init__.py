import logging
import os
import sys

import argparse
import numpy as np
from mkgu.assemblies import NeuroidAssembly

from neural_nlp.models.implementations import _model_mappings, load_model
from neural_nlp.stimuli import _mappings, load_stimuli
from neural_nlp.utils import store

_logger = logging.getLogger(__name__)


class _Defaults(object):
    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output'))


@store()
def get_activations(model_name, stimulus_set_name):
    _logger.debug('Loading model')
    model = load_model(model_name)
    _logger.debug('Loading stimuli')
    stimuli = load_stimuli(stimulus_set_name)

    _logger.debug("Retrieving activations")
    activations = []
    for i, sentence in enumerate(stimuli, start=1):
        if i % 10 == 0:
            _logger.debug("Sentence {}/{} ({:.0f}%)".format(i, len(stimuli), 100 * i / len(stimuli)))
        sentence_activations = model([sentence])
        activations.append(sentence_activations)
    activations = np.concatenate(activations)

    num_neurons = activations.shape[1]
    _logger.debug('Converting to {}x{} assembly'.format(num_neurons, len(stimuli)))
    return NeuroidAssembly(activations.T,
                           coords={
                               'stimulus': stimuli,
                               'dataset.name': ('stimulus', [stimulus_set_name] * len(stimuli)),
                               'neuroid': list(range(num_neurons)),
                               'model.name': ('neuroid', [model_name] * num_neurons),
                           },
                           dims=['neuroid', 'stimulus'])


def main():
    parser = argparse.ArgumentParser('model activations')
    parser.add_argument('--model', type=str, required=True, choices=list(_model_mappings.keys()))
    parser.add_argument('--dataset', type=str, required=True, choices=list(_mappings.keys()))
    parser.add_argument('--output_directory', type=str, default=_Defaults.output_directory)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    _logger.info("Running with args %s", vars(args))

    activations = get_activations(args.model, args.dataset)
    _logger.info("Activations computed: {}".format(activations))


if __name__ == '__main__':
    main()
