import logging

import networkx as nx

logger = logging.getLogger(__name__)


def combine_graph(graph, region_layer_mapping):
    layer_region_mapping = {layer: region for region, layers in region_layer_mapping.items() for layer in layers}
    combined_graph = nx.DiGraph()
    for region, layers in region_layer_mapping.items():
        if len(layers) == 0:
            continue

        for node in layers:
            for incoming_edge in graph.in_edges(node):
                source_node = incoming_edge[0]
                if source_node not in layers:
                    source_region = layer_region_mapping[source_node]
                    combined_graph.add_edge(source_region, region)

    return combined_graph


def cut_graph(graph, keep_nodes, fill_in=False):
    _cut_graph = graph.copy()
    remove_nodes = set(_cut_graph.nodes) - set(keep_nodes)
    for node in remove_nodes:
        if fill_in:
            in_edges = _cut_graph.in_edges(node)
            out_edges = _cut_graph.out_edges(node)
            if len(in_edges) > 1:
                raise NotImplementedError()
            if len(out_edges) > 1:
                raise NotImplementedError()
            if len(in_edges) > 0 and len(out_edges) > 0:
                _cut_graph.add_edge(next(iter(in_edges))[0], next(iter(out_edges))[1])

        _cut_graph.remove_node(node)
    return _cut_graph


def score_edge_ratio(comparison_graph, target_graph):
    unmatched = [edge for edge in target_graph.edges if edge not in comparison_graph.edges]
    return 1 - len(unmatched) / len(target_graph.edges)
