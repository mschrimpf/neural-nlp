import argparse
import logging
import os
import sys

from neural_metrics.data.nlp import data_mappings
from neural_metrics.models.nlp.implementations import model_mappings
from neural_metrics.utils import StorageCache

_logger = logging.getLogger(__name__)


class _Defaults(object):
    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output'))
    model_weights = True


class ActivationsWorker(object):
    def __init__(self, model_name, model_weights=_Defaults.model_weights,
                 output_directory=_Defaults.output_directory):
        self._logger = logging.getLogger(__class__.__name__ + "({})".format(model_name))
        self._output_directory = output_directory
        # model
        if not model_weights:
            raise NotImplementedError()
        self._model_name = model_name
        self._model_weights = model_weights
        self._logger.debug('Creating model')
        self._model = model_mappings[model_name]()
        # cache
        self._cache = StorageCache(self.get_savepath())

    def __call__(self, dataset_name):
        if dataset_name not in self._cache:
            self._logger.debug('Computing activations for dataset {}'.format(dataset_name))
            self._data = data_mappings[dataset_name]()
            activations = self._model(self._data)
            self._cache[dataset_name] = activations
        return self._cache[dataset_name]

    def get_savepath(self):
        return os.path.join(self._output_directory,
                            '{}-weights_{}-activations.pkl'.format(self._model_name, self._model_weights))


def main():
    parser = argparse.ArgumentParser('model activations')
    parser.add_argument('--model', type=str, required=True, choices=list(model_mappings.keys()))
    parser.add_argument('--dataset', type=str, required=True, choices=list(data_mappings.keys()))
    parser.add_argument('--output_directory', type=str, default=_Defaults.output_directory)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    _logger.info("Running with args %s", vars(args))

    activations_worker = ActivationsWorker(args.model, output_directory=args.output_directory)
    activations_worker(args.dataset)
    _logger.info("Saved to {}".format(activations_worker.get_savepath()))


if __name__ == '__main__':
    main()
