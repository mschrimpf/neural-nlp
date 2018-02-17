import unittest

import networkx as nx

from neural_metrics.metrics import model_graph
from neural_metrics.models import vgg16
from tests.metrics.anatomy import TestGraphComparison


class TestModelGraph(TestGraphComparison):
    def test_vgg16(self):
        model = vgg16(224)[0]
        layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool', 'fc1', 'fc2']
        target_graph = nx.DiGraph()
        for layer1, layer2 in zip(layers, layers[1:]):
            target_graph.add_edge(layer1, layer2)

        graph = model_graph(model, layers=layers)
        self.assert_graphs_equal(graph, target_graph)


if __name__ == '__main__':
    unittest.main()
