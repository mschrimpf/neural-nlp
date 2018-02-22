import unittest
from llist import dllist

from neural_metrics.metrics import _mapping_update_ranking, _mapping_update_all_surround


class TestMappingUpdateRanking(unittest.TestCase):
    def test_next_layer(self):
        def similarities(region, layers):
            return 0.4 if region == 'region1' and len(layers) == 1 and layers[0] == 'layer4' else 0.1

        linked_layers = dllist(['layer{}'.format(i + 1) for i in range(8)])
        mapping = {'region1': (('layer3',), 0.5), 'region2': (('layer6',), 0.5)}
        (region, layers), score = _mapping_update_ranking(linked_layers, mapping, similarities)
        self.assertEqual('region1', region)
        self.assertListEqual(['layer3', 'layer4'], layers)
        self.assertEqual(0.1, score)

    def test_after_next_layer(self):
        def similarities(region, layers):
            return 0.4 if region == 'region1' and len(layers) == 1 and layers[0] == 'layer5' else 0.1

        linked_layers = dllist(['layer{}'.format(i + 1) for i in range(8)])
        mapping = {'region1': (('layer3',), 0.5), 'region2': (('layer6',), 0.5)}
        (region, layers), score = _mapping_update_ranking(linked_layers, mapping, similarities)
        self.assertEqual('region1', region)
        self.assertListEqual(['layer3', 'layer4', 'layer5'], layers)
        self.assertEqual(0.1, score)

    def test_region_overlap(self):
        def similarities(region, layers):
            if region == 'region1' and len(layers) == 1:
                if layers[0] == 'layer7':
                    return 0.4
                elif layers[0] == 'layer4':
                    return 0.3
                else:
                    return 0.1
            else:
                return 0.1

        linked_layers = dllist(['layer{}'.format(i + 1) for i in range(8)])
        mapping = {'region1': (('layer3',), 0.5), 'region2': (('layer6',), 0.5)}
        (region, layers), score = _mapping_update_ranking(linked_layers, mapping, similarities)
        self.assertEqual('region1', region)
        self.assertListEqual(['layer3', 'layer4'], layers)
        self.assertEqual(0.1, score)

    def test_no_possibility(self):
        def similarities(region, layers):
            return 0.1

        linked_layers = dllist(['layer{}'.format(i + 1) for i in range(3)])
        mapping = {'region1': (('layer1', 'layer2'), 0.5), 'region2': (('layer3',), 0.5)}
        proposal = _mapping_update_ranking(linked_layers, mapping, similarities)
        self.assertIsNone(proposal)


class TestMappingAllSurrounding(unittest.TestCase):
    def test_choose_region(self):
        def similarities(region, layers):
            return 0.4 if region == 'region1' and set(layers) == {'layer2', 'layer3', 'layer4'} else 0.1

        linked_layers = dllist(['layer{}'.format(i + 1) for i in range(8)])
        mapping = {'region1': (('layer3',), 0.5), 'region2': (('layer6',), 0.5)}
        (region, layers), score = _mapping_update_all_surround(linked_layers, mapping, similarities)
        self.assertEqual('region1', region)
        self.assertListEqual(['layer2', 'layer3', 'layer4'], layers)
        self.assertEqual(0.4, score)


if __name__ == '__main__':
    unittest.main()
