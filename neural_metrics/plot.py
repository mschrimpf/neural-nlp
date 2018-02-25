import argparse
import logging
import pickle
import sys
from collections import OrderedDict, defaultdict

import os
from matplotlib import pyplot

from neural_metrics import metrics
from neural_metrics.metrics.physiology import _layers_correlation_meanstd

_logger = logging.getLogger(__name__)

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

type_color_mapping = {metrics.Type.PHYSIOLOGY: '#80b1d3', metrics.Type.ANATOMY: '#8dd3c7'}
area_color_mapping = {'V4': '#00cc66', 'IT': '#ffcc00'}


def plot_scores(scores, output_filepath=None):
    x = range(len(scores))
    pyplot.bar(x, [score.y for score in scores], yerr=[score.yerr for score in scores],
               tick_label=[score.name if score.type != metrics.Type.ANATOMY else 'anatomy' for score in scores],
               color=[type_color_mapping[score.type] for score in scores])
    _show_or_save(output_filepath)


def plot_layer_correlations(filepath, reverse=False, output_filepath=None):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    layer_metrics = data['data']

    region_layer_metric = defaultdict(OrderedDict)
    for (layer, region, variance), metric in layer_metrics.items():
        if len(layer) == 1:
            region_layer_metric[region][layer] = metric

    for region, layer_metric in region_layer_metric.items():
        if reverse:
            layer_metric = OrderedDict(reversed(list(layer_metric.items())))

        layers = list(layer_metric.keys())
        means, stds = _layers_correlation_meanstd(layer_metric)
        x = range(len(layer_metric))
        pyplot.errorbar(x, means, yerr=stds, label=region)
    pyplot.xticks(x, layers, rotation='vertical')
    pyplot.legend()

    _show_or_save(output_filepath)


def arrowed_spines(ax=None, arrow_length=10, labels=('', ''), arrowprops=None):
    """
    from https://gist.github.com/joferkington/3845684
    """
    xlabel, ylabel = labels
    ax = ax or pyplot.gca()
    if arrowprops is None:
        arrowprops = dict(arrowstyle='<|-', facecolor='black')

    for i, spine in enumerate(['left', 'bottom']):
        # Set up the annotation parameters
        t = ax.spines[spine].get_transform()
        xy, xycoords = [0.99, 0], ('axes fraction', t)
        xytext, textcoords = [arrow_length, 0], ('offset points', t)
        ha, va = 'left', 'bottom'

        if spine is 'bottom':
            xarrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext,
                                 textcoords=textcoords, ha=ha, va='center',
                                 arrowprops=arrowprops)
        else:
            yarrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1],
                                 xytext=xytext[::-1], textcoords=textcoords[::-1],
                                 ha='center', va=va, arrowprops=arrowprops)
    return xarrow, yarrow


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
