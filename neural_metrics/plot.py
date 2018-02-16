import argparse
import logging
import pickle
import sys
from collections import OrderedDict, Iterable

from matplotlib import pyplot

from neural_metrics.metrics.physiology import layers_correlation_meanstd

logger = logging.getLogger(__name__)


def plot_layer_correlations(filepaths, labels=None, reverse=False, output_filepath=None):
    if not isinstance(filepaths, Iterable):
        filepaths = [filepaths]
    for i, filepath in enumerate(filepaths):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        layer_metrics, args = data['layer_metrics'], data['args']
        if isinstance(args, argparse.Namespace): args = vars(args)
        default_label = '{} ({})'.format(args['region'], args['variance'])
        label = labels[i] if labels is not None else default_label
        if reverse:
            layer_metrics = OrderedDict(reversed(list(layer_metrics.items())))

        means, stds = layers_correlation_meanstd(layer_metrics)
        x = range(len(layer_metrics))
        pyplot.errorbar(x, means, yerr=stds, label=label)
        pyplot.xticks(x, layer_metrics.keys(), rotation='vertical')
    pyplot.legend()

    if output_filepath is None:
        pyplot.show()
    else:
        pyplot.savefig(output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--correlations_filepaths', type=str, nargs='+', required=True)

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))
    plot_layer_correlations(args.correlations_filepaths)


if __name__ == '__main__':
    main()
