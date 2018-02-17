import argparse
import logging
import sys
import unittest

import networkx as nx

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    logger.info("Running with args %s", vars(args))


if __name__ == '__main__':
    main()


class TestGraphComparison(unittest.TestCase):
    def assert_graphs_equal(self, graph1, graph2):
        self.assertListEqual(list(graph1.nodes), list(graph2.nodes),
                             "Graph nodes are not equal: {} <> {}".format(graph1.nodes, graph2.nodes))
        self.assertTrue(nx.is_isomorphic(graph1, graph2),
                        "Graphs are not isomorphic: {} <> {}".format(graph1.edges, graph2.edges))