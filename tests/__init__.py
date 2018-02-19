import unittest

import networkx as nx


class TestGraphComparison(unittest.TestCase):
    def assert_graphs_equal(self, graph1, graph2):
        self.assertListEqual(list(graph1.nodes), list(graph2.nodes),
                             "Graph nodes are not equal: {} <> {}".format(graph1.nodes, graph2.nodes))
        self.assertTrue(nx.is_isomorphic(graph1, graph2),
                        "Graphs are not isomorphic: {} <> {}".format(graph1.edges, graph2.edges))
