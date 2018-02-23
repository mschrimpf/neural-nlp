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
from neural_metrics.models import model_name_from_activations_filepath

logger = logging.getLogger(__name__)


def _mapping_update_prevnext(linked_layers, mapping, similarities, ignored_regions=()):
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
    mapped_layers = get_mapped_layers(mapping)
    candidates = []
    for region, (layers, score) in mapping.items():
        if region in ignored_regions:
            continue
        for prev_next in [(0, 'prev'), (-1, 'next')]:
            linked_node = _linked_node(linked_layers, layers[prev_next[0]])
            prev_next_node = getattr(linked_node, prev_next[1])
            if prev_next_node is not None and prev_next_node.value not in mapped_layers:
                layers_candidate = ((prev_next_node.value,) + layers) if prev_next[1] == 'prev' \
                    else layers + (prev_next_node.value,)
                candidates.append({'layers': layers_candidate, 'region': region})
    if len(candidates) == 0:
        return None
    candidate_similarities = {(candidate['region'], candidate['layers']):
                                  similarities(layers=candidate['layers'], region=candidate['region'])
                              for candidate in candidates}
    mapping_update = max(candidate_similarities.items(), key=itemgetter(1))
    return mapping_update


def _mapping_update_ranking(linked_layers, mapping, similarities, ignored_regions=()):
    layer_region_mapping = {layer: region for region, (layers, score) in mapping.items() for layer in layers}

    def enclosed_layers(layers, boundary_layer):
        linked_node = _linked_node(linked_layers, boundary_layer)
        for prevnext in ('prev', 'next'):
            node = linked_node
            enclosed, match = [], False
            while node is not None:
                match = match or node.value in layers
                if match and node.value not in layers:
                    return enclosed
                if prevnext == 'prev':
                    enclosed.insert(0, node.value)
                else:
                    enclosed.append(node.value)
                node = getattr(node, prevnext)
            if match:  # end of list
                return enclosed
        assert False

    mapped_layers = get_mapped_layers(mapping)
    candidate_single_layer_scores = []
    for region in mapping.keys():
        if region in ignored_regions:
            continue
        region_candidate_scores = [({'region': region, 'layer': layer}, similarities(region=region, layers=[layer]))
                                   for layer in linked_layers if layer not in mapped_layers
                                   and layer_connected_to_region(layer, region, linked_layers, layer_region_mapping)]
        candidate_single_layer_scores.extend(region_candidate_scores)
    if len(candidate_single_layer_scores) == 0:
        return None
    ranked_layers = sorted(candidate_single_layer_scores, key=itemgetter(1), reverse=True)
    next_best_layer = ranked_layers[0][0]
    region, layer = next_best_layer['region'], next_best_layer['layer']
    layers = enclosed_layers(mapping[region][0], layer)
    return (region, layers), similarities(region=region, layers=layers)


def _mapping_update_all_surround(linked_layers, mapping, similarities, ignored_regions=()):
    layer_region_mapping = {layer: region for region, (layers, score) in mapping.items() for layer in layers}

    candidates = []
    for region, (layer_basis, _) in mapping.items():
        if region in ignored_regions:
            continue
        expanded_layers = [layer for layer in linked_layers
                           if (layer not in layer_region_mapping or layer_region_mapping[layer] == region)
                           and layer_connected_to_region(layer, region, linked_layers, layer_region_mapping)]
        candidate_layers = [expanded_layers[start:start + num_layers]
                            for num_layers in range(1, len(expanded_layers) + 1)
                            for start in range(len(expanded_layers) - num_layers)]
        candidate_layers = [layers for layers in candidate_layers if all(layer in layers for layer in layer_basis)
                            and set(layers) != set(layer_basis)]
        for layers in candidate_layers:
            candidates.append({'region': region, 'layers': layers,
                               'score': similarities(region=region, layers=layers)})
    if len(candidates) == 0:
        return None
    ranked_candidates = sorted(candidates, key=lambda candidate: candidate['score'], reverse=True)
    mapping_update = ranked_candidates[0]
    return (mapping_update['region'], mapping_update['layers']), mapping_update['score']


def layer_connected_to_region(layer, region, linked_layers, layer_region_mapping):
    """
    Path from layer to region must not be interrupted by another region
    """
    if layer in layer_region_mapping and layer_region_mapping[layer] == region:
        return True  # layer is IN region

    linked_node = _linked_node(linked_layers, layer)
    num_foreign_regions = 0
    for prevnext in ('prev', 'next'):
        node = getattr(linked_node, prevnext)
        while node is not None:
            if node.value in layer_region_mapping:
                foreign_region = layer_region_mapping[node.value] != region
                num_foreign_regions = num_foreign_regions + foreign_region
                break
            node = getattr(node, prevnext)
        # for the sake of this algorithm, we consider the end of the nodes to be a foreign region
        num_foreign_regions = num_foreign_regions + (node is None)
    return num_foreign_regions <= 1


def physiology_mapping(model_activations_filepath, regions,
                       map_all_layers=True, _mapping_update=_mapping_update_all_surround,
                       no_negative_updates=False, use_cached=True):
    similarities = SimilarityWorker(model_activations_filepath, regions, use_cached=use_cached)
    assert len(similarities.get_model_layers()) > len(regions)

    mapping = map_single_layers(regions, similarities)
    if not map_all_layers:
        return mapping

    # all layers
    linked_layers = dllist(similarities.get_model_layers())
    mapping = OrderedDict((region, ((layer,), score)) for region, (layer, score) in mapping.items())
    finished_regions = []
    while len(similarities.get_model_layers()) != len(get_mapped_layers(mapping)):
        mapping_update = _mapping_update(linked_layers, mapping, similarities, ignored_regions=finished_regions)
        if mapping_update is None:
            logger.info("No more mapping proposals")
            break
        (region, layers), score = mapping_update
        if score < mapping[region][1]:
            logger.warning("Negative change from candidate in region {} (was {}: {:.4f}, now {}: {:.4f}){}".format(
                region, ",".join(mapping[region][0]), mapping[region][1], ",".join(layers), score,
                "ignoring region" if no_negative_updates else ""))
            if no_negative_updates:
                finished_regions.append(region)
        mapping[region] = layers, score
        logger.debug("Update mapping: {} -> {} ({:.2f})".format(region, ",".join(layers), score))
    return mapping


def map_single_layers(regions, similarities):
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
    return mapping


def score_anatomy(model, region_layer_score_mapping):
    _model_graph = model_graph(model, layers=get_mapped_layers(region_layer_score_mapping))
    _model_graph = combine_graph(_model_graph, OrderedDict((region, layers) for region, (layers, score)
                                                           in region_layer_score_mapping.items()))
    return score_edge_ratio(_model_graph, relevant_regions=region_layer_score_mapping.keys())


def get_mapped_layers(region_layer_score_mapping):
    return [*itertools.chain(*[layers for layers, score in region_layer_score_mapping.values()])]


def _linked_node(linked_layers, node_value):
    return linked_layers.nodeat([node == node_value for node in linked_layers].index(True))


class Score(object):
    def __init__(self, name, type, y, yerr, explanation):
        self.name = name
        self.type = type
        self.y = y
        self.yerr = yerr
        self.explanation = explanation


def score_model_activations(activations_filepath, regions, model_name=None, map_all_layers=True, use_cached=True):
    model_name = model_name or model_name_from_activations_filepath(activations_filepath)
    model = models.model_mappings[model_name](image_size=models._Defaults.image_size)[0]

    region_layer_mapping = physiology_mapping(activations_filepath, regions, map_all_layers=map_all_layers)
    logger.info("Physiology mapping: " + ", ".join(
        "{} -> {} ({:.2f})".format(region, ",".join(layers), score)
        for region, (layers, score) in region_layer_mapping.items()))

    return [Score(name=region, type='physiology', y=score, yerr=0, explanation=layers)
            for region, (layers, score) in region_layer_mapping.items()]

    anatomy_score = score_anatomy(model, region_layer_mapping)
    logger.info("Anatomy score: {}".format(anatomy_score))

    return [Score(name=region, type='physiology', y=score, yerr=0, explanation=layers)
            for region, (layers, score) in region_layer_mapping.items()] + \
           [Score(name='edge_ratio', type='anatomy', y=anatomy_score, yerr=0, explanation=None)]


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

    scores = score_model_activations(args.activations_filepath, regions=args.regions, model_name=args.model,
                                     map_all_layers=args.map_all_layers)
    print(scores)


if __name__ == '__main__':
    main()
