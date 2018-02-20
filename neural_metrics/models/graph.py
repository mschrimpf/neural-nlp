import logging

import networkx as nx

from neural_metrics.models.type import get_model_type, ModelType

_logger = logging.getLogger(__name__)


def get_model_graph(model):
    model_type = get_model_type(model)
    if model_type == ModelType.KERAS:
        return _get_model_graph_keras(model)
    elif model_type == ModelType.PYTORCH:
        return _get_model_graph_pytorch(model)


def _get_model_graph_keras(model):
    g = nx.DiGraph()
    for layer in model.layers:
        for outbound_node in layer._outbound_nodes:
            g.add_edge(layer.name, outbound_node.outbound_layer.name)
    return g


def _get_model_graph_pytorch(model):
    raise NotImplementedError()
