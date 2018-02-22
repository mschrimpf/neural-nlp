import argparse
import logging
import pickle
import sys
from collections import OrderedDict, Iterable

import os
from matplotlib import pyplot

from neural_metrics.metrics.physiology import layers_correlation_meanstd

_logger = logging.getLogger(__name__)

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def plot_scores(scores, output_filepath=None):
    type_color_mapping = {'physiology': '#80b1d3', 'anatomy': '#8dd3c7'}

    x = range(len(scores))
    pyplot.bar(x, [score.y for score in scores], yerr=[score.yerr for score in scores],
               tick_label=[score.name if score.type != 'anatomy' else 'anatomy' for score in scores],
               color=[type_color_mapping[score.type] for score in scores])
    _show_or_save(output_filepath)


def plot_layer_correlations(filepaths, labels=None, reverse=False, output_filepath=None):
    if not isinstance(filepaths, Iterable):
        filepaths = [filepaths]
    for i, filepath in enumerate(filepaths):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        layer_metrics, args = data['data'], data['args']
        if isinstance(args, argparse.Namespace): args = vars(args)
        default_label = '{} ({})'.format(args['region'], args['variance'])
        label = labels[i] if labels is not None else default_label
        if reverse:
            layer_metrics = OrderedDict(reversed(list(layer_metrics.items())))

        layer_metrics = OrderedDict((layer, scores) for layer, scores in layer_metrics.items() if len(layer) == 1)
        means, stds = layers_correlation_meanstd(layer_metrics)
        x = range(len(layer_metrics))
        pyplot.errorbar(x, means, yerr=stds, label=label)
        pyplot.xticks(x, layer_metrics.keys(), rotation='vertical')
    pyplot.legend()

    _show_or_save(output_filepath)


def _show_or_save(output_filepath):
    if output_filepath is None:
        pyplot.show()
    else:
        _logger.info('Plot saved to {}'.format(output_filepath))
        pyplot.savefig(output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--correlations_filepaths', type=str, nargs='+', required=True)

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running with args %s", vars(args))
    plot_layer_correlations(args.correlations_filepaths)


if __name__ == '__main__':
    main()
