from neural_metrics.models import get_model_graph_keras
from . import graphs
from .graphs import cut_graph, combine_graph
import networkx as nx

anatomy_graph = nx.DiGraph()  # derived from Felleman & van Essen
anatomy_graph.add_edge('input', 'V1')
anatomy_graph.add_edge('V1', 'V2')
anatomy_graph.add_edge('V1', 'V4')
anatomy_graph.add_edge('V2', 'V1')
anatomy_graph.add_edge('V2', 'V4')
anatomy_graph.add_edge('V4', 'V1')
anatomy_graph.add_edge('V4', 'V2')
anatomy_graph.add_edge('V4', 'IT')
anatomy_graph.add_edge('IT', 'V4')  # ignoring {p,c,a}IT subdivision for now


def score_edge_ratio(model_graph, relevant_regions):
    anatomy_subgraph = cut_graph(anatomy_graph, keep_nodes=relevant_regions)
    return graphs.score_edge_ratio(model_graph, anatomy_subgraph)


def model_graph(model, layers):
    graph = get_model_graph_keras(model)
    return cut_graph(graph, keep_nodes=layers, fill_in=True)
