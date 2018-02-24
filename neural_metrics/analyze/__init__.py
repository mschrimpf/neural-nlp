import logging
import os
from llist import dllist

import numpy as np
import sys
from matplotlib import pyplot

from neural_metrics import models
from neural_metrics.metrics import map_single_layers, _linked_node
from neural_metrics.metrics.physiology import SimilarityWorker

_logger = logging.getLogger(__name__)


def plot_layer_combinations_from_single_layer(model, model_weights=models._Defaults.model_weights,
                                              regions=('V4', 'IT')):
    activations_filepath = models.get_savepath(model, model_weights=model_weights)
    assert os.path.isfile(activations_filepath)
    for region in regions:
        similarities = SimilarityWorker(activations_filepath, (region,))
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


def plot_all_connected_layer_combinations(model, model_weights=models._Defaults.model_weights,
                                          regions=('V4', 'IT'), cutoff=20):
    activations_filepath = models.get_savepath(model, model_weights=model_weights)
    assert os.path.isfile(activations_filepath)
    for region in regions:
        similarities = SimilarityWorker(activations_filepath, (region,))
        layers = similarities.get_model_layers()
        layer_combinations = [layers[start:start + num_layers]
                              for num_layers in range(1, len(layers) + 1)
                              for start in range(len(layers) - num_layers + 1)]
        _logger.debug("{} combinations".format(len(layer_combinations)))
        scores = [similarities(region=region, layers=layers) for layers in layer_combinations]
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:cutoff]

        pyplot.plot(range(len(ranked_indices)), [scores[i] for i in ranked_indices])
        pyplot.xticks(range(len(ranked_indices)),
                      ["{}{}".format(layers[0], " - " + layers[-1] if len(layers) > 1 else "")
                       for layers in [layer_combinations[i] for i in ranked_indices]], rotation='vertical')
        pyplot.title(region)
        pyplot.show()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName("DEBUG"))
    plot_all_connected_layer_combinations('resnet50')
