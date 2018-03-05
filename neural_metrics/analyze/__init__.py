import logging
import os
import sys
from llist import dllist

import numpy as np
from matplotlib import pyplot

from neural_metrics import models
from neural_metrics.metrics.physiology import SimilarityWorker
from neural_metrics.metrics.physiology.mapping import map_single_layers, _linked_node
from neural_metrics.models import ActivationsWorker

_logger = logging.getLogger(__name__)


def plot_layer_combinations_from_single_layer(model_name, layers, model_weights=models._Defaults.model_weights,
                                              regions=('V4', 'IT')):
    activations_worker = ActivationsWorker(model_name=model_name, model_weights=model_weights)
    model_activations = activations_worker(layers=layers)
    activations_filepath = activations_worker.get_savepath()
    for region in regions:
        similarities = SimilarityWorker(model_activations, basepath=activations_filepath, regions=(region,))
        single_layer = map_single_layers((region,), similarities)[region][0]
        linked_layers = dllist(similarities.get_model_layers())
        linked_node = _linked_node(linked_layers, single_layer)

        layer_combinations = [[single_layer]]
        prev_node = linked_node.prev
        while prev_node is not None:
            layer_combinations.insert(0, [prev_node.value] + layer_combinations[0])
            prev_node = prev_node.prev
        next_node = linked_node.next
        while next_node is not None:
            layer_combinations.append(layer_combinations[-1] + [next_node.value])
            next_node = next_node.next
        _logger.debug("Layer combinations {}".format(layer_combinations))

        scores = [similarities(region=region, layers=layers) for layers in layer_combinations]
        _logger.debug("Scores {}".format(scores))

        pyplot.plot(range(len(layer_combinations)), scores)
        initial_pos = layer_combinations.index([single_layer])
        pyplot.plot(initial_pos, scores[initial_pos], marker='o', markersize=20)
        max_pos = np.argmax(scores)
        pyplot.plot(max_pos, scores[max_pos], marker='*', markersize=20)

        pyplot.xticks(range(len(layer_combinations)),
                      ["{}{}".format(layers[0], " - " + layers[-1] if len(layers) > 1 else "")
                       for layers in layer_combinations], rotation='vertical')
        pyplot.title(region)
        pyplot.show()


def plot_all_connected_layer_combinations(model_name, layers, model_weights=models._Defaults.model_weights,
                                          regions=('V4', 'IT'), cutoff=10):
    activations_worker = ActivationsWorker(model_name=model_name, model_weights=model_weights)
    model_activations = activations_worker(layers=layers)
    activations_filepath = activations_worker.get_savepath()
    assert os.path.isfile(activations_filepath)
    for region in regions:
        _logger.debug("Region {}".format(region))
        similarities = SimilarityWorker(model_activations, basepath=activations_filepath, regions=(region,))
        layers = similarities.get_model_layers()
        single_layer_scores = [(layer, similarities(region=region, layers=[layer])) for layer in layers]
        best_single_layer, best_single_score = max(single_layer_scores, key=lambda layer_score: layer_score[1])
        layer_combinations = [layers[start:start + num_layers]
                              for num_layers in range(1, len(layers) + 1)
                              for start in range(len(layers) - num_layers + 1)]
        _logger.debug("{} combinations".format(len(layer_combinations)))
        scores = [similarities(region=region, layers=layers) for layers in layer_combinations]
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        x = list(range(len(ranked_indices)))
        y = [scores[i] for i in ranked_indices]

        f, axes = pyplot.subplots(1, 2 if len(ranked_indices) > cutoff * 2 else 1, sharey=True, facecolor='w')
        pyplot.suptitle(region)
        for ax in axes:
            ax.plot(x, y)
            ax.plot(ax.get_xlim(), [best_single_score, best_single_score], color='gray', linestyle='dashed')
            ax.set_xticks(x)
            ax.set_xticklabels(["{}{}".format(layers[0], " - " + layers[-1] if len(layers) > 1 else "")
                                for layers in [layer_combinations[i] for i in ranked_indices]], rotation='vertical')
        axes[-1].text(x[-cutoff], best_single_score, 'best single layer ({})'.format(best_single_layer))
        if len(axes) > 1:
            axes[0].set_xlim(0, cutoff)
            axes[1].set_xlim(len(ranked_indices) - cutoff, len(ranked_indices))

            cutout_axes(*axes)

        pyplot.tight_layout()
        pyplot.show()


def cutout_axes(ax1, ax2):
    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    # cut-out lines
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
