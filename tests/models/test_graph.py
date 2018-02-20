import unittest

import networkx as nx

from neural_metrics.models.graph import get_model_graph
from neural_metrics.models.implementations import vgg16
from tests import TestGraphComparison


class TestModelGraph(TestGraphComparison):
    def test_keras_vgg16(self):
        model = vgg16(224)[0]
        layers = ['input_1',
                  'block1_conv1', 'block1_conv2', 'block1_pool',
                  'block2_conv1', 'block2_conv2', 'block2_pool',
                  'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
                  'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
                  'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool',
                  'flatten', 'fc1', 'fc2', 'predictions']
        target_graph = nx.DiGraph()
        for layer1, layer2 in zip(layers, layers[1:]):
            target_graph.add_edge(layer1, layer2)

        graph = get_model_graph(model)
        self.assert_graphs_equal(graph, target_graph)


if __name__ == '__main__':
    unittest.main()
