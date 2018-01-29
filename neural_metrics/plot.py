import pickle
from collections import OrderedDict, Iterable

from matplotlib import pyplot

from neural_metrics.compare import layers_correlation_meanstd


def plot_layer_correlations(filepaths, reverse=False):
    if not isinstance(filepaths, Iterable):
        filepaths = [filepaths]
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        layer_metrics, args = data['layer_metrics'], data['args']
        if reverse:
            layer_metrics = OrderedDict(reversed(list(layer_metrics.items())))
        means, stds = layers_correlation_meanstd(layer_metrics)

        x = range(len(layer_metrics))
        pyplot.errorbar(x, means, yerr=stds, label='%s %s' % (args.region, args.variance))
        pyplot.xticks(x, layer_metrics.keys(), rotation='vertical')
    pyplot.legend()
