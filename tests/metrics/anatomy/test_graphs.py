import unittest
from collections import OrderedDict

import networkx as nx

from neural_metrics.metrics.anatomy.graphs import combine_graph, score_edge_ratio, cut_graph
from tests.metrics.anatomy import TestGraphComparison


class TestCombineGraph(TestGraphComparison):
    def test_unary_mapping(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        mapping = OrderedDict([('1', ['A']), ('2', ['B'])])
        combined_graph = combine_graph(graph, mapping)

        target_graph = nx.DiGraph()
        target_graph.add_edge('1', '2')
        self.assert_graphs_equal(combined_graph, target_graph)

    def test_combine_end(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        mapping = OrderedDict([('1', ['A']), ('2', ['B', 'C'])])
        combined_graph = combine_graph(graph, mapping)

        target_graph = nx.DiGraph()
        target_graph.add_edge('1', '2')
        self.assert_graphs_equal(combined_graph, target_graph)

    def test_combine_skip(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        graph.add_edge('A', 'D')
        mapping = OrderedDict([('1', ['A']), ('2', ['B']), ('3', ['C', 'D'])])
        combined_graph = combine_graph(graph, mapping)

        target_graph = nx.DiGraph()
        target_graph.add_edge('1', '2')
        target_graph.add_edge('2', '3')
        target_graph.add_edge('1', '3')
        self.assert_graphs_equal(combined_graph, target_graph)


class TestCutGraph(TestGraphComparison):
    def test_cut_nothing(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        _cut_graph = cut_graph(graph, keep_nodes=['A', 'B'])
        self.assert_graphs_equal(_cut_graph, graph)

    def test_2nodes(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        _cut_graph = cut_graph(graph, keep_nodes=['A'])

        target_graph = nx.DiGraph()
        target_graph.add_node('A')
        self.assert_graphs_equal(_cut_graph, target_graph)

    def test_2nodes_fill_in(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        _cut_graph = cut_graph(graph, keep_nodes=['A'], fill_in=True)

        target_graph = nx.DiGraph()
        target_graph.add_node('A')
        self.assert_graphs_equal(_cut_graph, target_graph)

    def test_4nodes(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        graph.add_edge('A', 'D')
        graph.add_edge('B', 'C')
        _cut_graph = cut_graph(graph, keep_nodes=['B', 'C', 'D'])

        target_graph = nx.DiGraph()
        target_graph.add_edge('B', 'C')
        target_graph.add_edge('C', 'D')
        target_graph.add_edge('B', 'C')
        self.assert_graphs_equal(_cut_graph, target_graph)

    def test_4nodes_fill_in(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        graph.add_edge('C', 'D')
        _cut_graph = cut_graph(graph, keep_nodes=['A', 'D'], fill_in=True)

        target_graph = nx.DiGraph()
        target_graph.add_edge('A', 'D')
        self.assert_graphs_equal(_cut_graph, target_graph)


class TestScoreEdgeRatio(unittest.TestCase):
    def test_2nodes_equal(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        self.assertEqual(1, score_edge_ratio(graph, graph))

    def test_2nodes_different(self):
        graph1 = nx.DiGraph()
        graph1.add_edge('A', 'B')
        graph2 = nx.DiGraph()
        graph2.add_edge('B', 'A')
        self.assertEqual(0, score_edge_ratio(graph1, graph2))

    def test_3nodes_equal(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        self.assertEqual(1, score_edge_ratio(graph, graph))

    def test_3nodes_1edge_missing(self):
        graph1 = nx.DiGraph()
        graph1.add_edge('A', 'B')
        graph1.add_edge('B', 'C')
        graph2 = nx.DiGraph()
        graph2.add_edge('A', 'B')
        graph2.add_edge('C', 'B')
        self.assertEqual(0.5, score_edge_ratio(graph1, graph2))


if __name__ == '__main__':
    unittest.main()
