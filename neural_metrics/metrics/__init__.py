import argparse
import logging
import os
import sys
from collections import OrderedDict
from operator import itemgetter

import itertools
from llist import dllist

from neural_metrics.metrics.physiology import SimilarityWorker, load_model_activations, layers_from_raw_activations

logger = logging.getLogger(__name__)


def physiology_mapping(model_activations, regions, map_all_layers=True):
    """
    Pseudocode:

    ```
    input: layers L, target regions R; output: mapping M from R to L
    map each single layer in L to each region in R, add the best single layers to M
    while not each layer mapped:
      for mapping m in M:
        compute improvement in m if we add one more layer above and below m
      choose the best single layer addition improvement and add to the corresponding m
    ```
    """
    similarities = SimilarityWorker(model_activations, regions)

    # single layer
    single_layer_similarities = {(layer, region): similarities(layers=layer, region=region)
                                 for layer in layers_from_raw_activations(model_activations) for region in regions}
    mapping = OrderedDict()
    for region in regions:
        best_layer = max([layer for layer, _region in single_layer_similarities.keys() if _region == region],
                         key=lambda layer: single_layer_similarities[(layer, region)])
        mapping[region] = best_layer
    if not map_all_layers:
        return mapping

    # all layers
    linked_layers = dllist(layers_from_raw_activations(model_activations))
    mapping = OrderedDict((region, [layer]) for region, layer in zip(mapping.keys(), linked_layers))
    while len(layers_from_raw_activations(model_activations)) != len([*(itertools.chain(*mapping.values()))]):
        candidates = []
        for region, layers in mapping.items():
            for prev_next in [(0, 'prev'), (-1, 'next')]:
                linked_node = linked_layers.nodeat([node == layers[prev_next[0]] for node in linked_layers].index(True))
                prev_next_node = getattr(linked_node, prev_next[1])
                if prev_next_node is not None and prev_next_node.value not in itertools.chain(*mapping.values()):
                    candidates.append({'layers': tuple(layers + [prev_next_node.value]), 'region': region})
        candidate_similarities = {(candidate['region'], candidate['layers']):
                                      similarities(layers=candidate['layers'], region=candidate['region'])
                                  for candidate in candidates}
        best_candidate = max(candidate_similarities.items(), key=itemgetter(1))[0]
        region, layers = best_candidate[0], best_candidate[1]
        mapping[region] = layers
        logger.debug("Update mapping: {} -> {}".format(region, ",".join(layers)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_filepath', type=str, nargs='+',
                        default=[os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sorted', 'Chairs',
                                              'vgg16-activations.pkl')],
                        help='one or more filepaths to the model activations')
    parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'], help='region(s) in brain to compare to')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))

    for activations_filepath in args.activations_filepath:
        model_activations = load_model_activations(activations_filepath)
        region_layer_mapping = physiology_mapping(model_activations, args.regions)


if __name__ == '__main__':
    main()
