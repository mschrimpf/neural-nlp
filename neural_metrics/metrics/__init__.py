import argparse
import itertools
import logging
import os
import sys
from collections import OrderedDict
from operator import itemgetter

from llist import dllist

from neural_metrics import models
from neural_metrics.metrics.anatomy import combine_graph, score_edge_ratio, model_graph
from neural_metrics.metrics.physiology import SimilarityWorker, load_model_activations, layers_from_raw_activations
from neural_metrics.models import model_from_activations_filepath

logger = logging.getLogger(__name__)


def physiology_mapping(model_activations_filepath, regions, map_all_layers=True):
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
    similarities = SimilarityWorker(model_activations_filepath, regions)
    assert len(similarities.get_model_layers()) > len(regions)

    # single layer
    single_layer_similarities = {(layer, region): similarities(layers=layer, region=region)
                                 for layer in similarities.get_model_layers() for region in regions}
    mapping = OrderedDict()
    for region in regions:
        best_layer = max([layer for layer, _region in single_layer_similarities.keys() if _region == region
                          # not mapped already
                          and layer not in [mapped_layer for mapped_layer, mapped_score in mapping.values()]],
                         key=lambda layer: single_layer_similarities[(layer, region)])
        score = single_layer_similarities[(best_layer, region)]
        mapping[region] = best_layer, score
        logger.debug("Update mapping: {} -> {} ({:.2f})".format(region, best_layer, score))
    if not map_all_layers:
        return mapping

    # all layers
    linked_layers = dllist(similarities.get_model_layers())
    mapping = OrderedDict((region, ((layer,), score)) for region, (layer, score) in mapping.items())
    while len(similarities.get_model_layers()) != len(get_mapped_layers(mapping)):
        mapped_layers = get_mapped_layers(mapping)
        candidates = []
        for region, (layers, score) in mapping.items():
            for prev_next in [(0, 'prev'), (-1, 'next')]:
                linked_node = linked_layers.nodeat([node == layers[prev_next[0]] for node in linked_layers].index(True))
                prev_next_node = getattr(linked_node, prev_next[1])
                if prev_next_node is not None and prev_next_node.value not in mapped_layers:
                    layers_candidate = ((prev_next_node.value,) + layers) if prev_next[1] == 'prev' \
                        else layers + (prev_next_node.value,)
                    candidates.append({'layers': layers_candidate, 'region': region})
        candidate_similarities = {(candidate['region'], candidate['layers']):
                                      similarities(layers=candidate['layers'], region=candidate['region'])
                                  for candidate in candidates}
        (region, layers), score = max(candidate_similarities.items(), key=itemgetter(1))
        if score < mapping[region][1]:
            logger.warning("Negative improvement from candidate in region {} (was {}: {:.4f}, now {}: {:.4f})".format(
                region, ",".join(mapping[region][0]), mapping[region][1], ",".join(layers), score))
        mapping[region] = layers, score
        logger.debug("Update mapping: {} -> {} ({:.2f})".format(region, ",".join(layers), score))
    return mapping


def score_anatomy(model, region_layer_score_mapping):
    _model_graph = model_graph(model, layers=get_mapped_layers(region_layer_score_mapping))
    _model_graph = combine_graph(_model_graph, OrderedDict((region, layers) for region, (layers, score)
                                                           in region_layer_score_mapping.items()))
    return score_edge_ratio(_model_graph, relevant_regions=region_layer_score_mapping.keys())


def get_mapped_layers(region_layer_score_mapping):
    return [*itertools.chain(*[layers for layers, score in region_layer_score_mapping.values()])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_filepath', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sorted',
                                                             'vgg16-weights_imagenet-activations.pkl')),
                        help='one or more filepaths to the model activations')
    parser.add_argument('--model', type=str, default=None, choices=models.model_mappings.keys(),
                        help='name of the model. Inferred from `--activations_filepath` if None')
    parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'], help='region(s) in brain to compare to')
    parser.add_argument('--map_all_layers', action='store_true', default=True)
    parser.add_argument('--no-map_all_layers', action='store_false', dest='map_all_layers')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args {}".format(vars(args)))

    model_name = args.model if args.model else model_from_activations_filepath(args.activations_filepath)
    model = models.model_mappings[model_name](image_size=models._Defaults.image_size)[0]

    region_layer_mapping = physiology_mapping(args.activations_filepath, args.regions,
                                              map_all_layers=args.map_all_layers)
    logger.info("Physiology mapping: " + ", ".join(
        "{} -> {} ({:.2f})".format(region, ",".join(layers), score)
        for region, (layers, score) in region_layer_mapping.items()))

    anatomy_score = score_anatomy(model, region_layer_mapping)
    logger.info("Anatomy score: {}".format(anatomy_score))


if __name__ == '__main__':
    main()
