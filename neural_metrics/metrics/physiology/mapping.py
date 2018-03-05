import itertools
import logging
from collections import OrderedDict
from llist import dllist
from operator import itemgetter

from . import SimilarityWorker

_logger = logging.getLogger(__name__)


def mapping_update_prevnext(linked_layers, mapping, similarities, ignored_regions=()):
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


def mapping_update_ranking(linked_layers, mapping, similarities, ignored_regions=()):
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


def mapping_update_all_surround(linked_layers, mapping, similarities, ignored_regions=()):
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
                            for start in range(len(expanded_layers) - num_layers + 1)]
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


def physiology_mapping(model_activations, basepath, regions,
                       map_all_layers=True, _mapping_update=mapping_update_all_surround,
                       no_negative_updates=True, output_directory=None, use_cached=True):
    similarities = SimilarityWorker(model_activations, basepath=basepath, regions=regions,
                                    output_directory=output_directory, use_cached=use_cached)
    assert len(similarities.get_model_layers()) >= len(regions)

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
            _logger.info("No more mapping proposals")
            break
        (region, layers), score = mapping_update
        if score < mapping[region][1]:
            _logger.warning("Negative change from candidate in region {} (was {}: {:.4f}, now {}: {:.4f}){}".format(
                region, ",".join(mapping[region][0]), mapping[region][1], ",".join(layers), score,
                " - ignoring region" if no_negative_updates else ""))
            if no_negative_updates:
                finished_regions.append(region)
        mapping[region] = layers, score
        _logger.debug("Update mapping: {} -> {} ({:.2f})".format(region, ",".join(layers), score))
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
        _logger.debug("Update mapping: {} -> {} ({:.2f})".format(region, best_layer, score))
    return mapping


def _linked_node(linked_layers, node_value):
    return linked_layers.nodeat([node == node_value for node in linked_layers].index(True))


def get_mapped_layers(region_layer_score_mapping):
    return [*itertools.chain(*[layers for layers, score in region_layer_score_mapping.values()])]
