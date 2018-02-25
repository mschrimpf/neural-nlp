import argparse
import logging
import os
import sys
from collections import OrderedDict

from neural_metrics import models
from neural_metrics.metrics.anatomy import combine_graph, score_edge_ratio, model_graph
from neural_metrics.metrics.physiology.mapping import physiology_mapping, get_mapped_layers
from neural_metrics.models import model_name_from_activations_filepath

_logger = logging.getLogger(__name__)


def score_anatomy(model, region_layer_score_mapping):
    _model_graph = model_graph(model, layers=get_mapped_layers(region_layer_score_mapping))
    _model_graph = combine_graph(_model_graph, OrderedDict((region, layers) for region, (layers, score)
                                                           in region_layer_score_mapping.items()))
    return score_edge_ratio(_model_graph, relevant_regions=region_layer_score_mapping.keys())


class Score(object):
    def __init__(self, name, type, y, yerr, explanation):
        self.name = name
        self.type = type
        self.y = y
        self.yerr = yerr
        self.explanation = explanation


def score_model_activations(activations_filepath, regions, model_name=None, map_all_layers=True):
    region_layer_mapping = physiology_mapping(activations_filepath, regions, map_all_layers=map_all_layers)
    _logger.info("Physiology mapping: " + ", ".join(
        "{} -> {} ({:.2f})".format(region, ",".join(layers), score)
        for region, (layers, score) in region_layer_mapping.items()))

    return [Score(name=region, type='physiology', y=score, yerr=0, explanation=layers)
            for region, (layers, score) in region_layer_mapping.items()]

    model_name = model_name or model_name_from_activations_filepath(activations_filepath)
    model = models.model_mappings[model_name](image_size=models._Defaults.image_size)[0]
    anatomy_score = score_anatomy(model, region_layer_mapping)
    _logger.info("Anatomy score: {}".format(anatomy_score))

    return [Score(name=region, type='physiology', y=score, yerr=0, explanation=layers)
            for region, (layers, score) in region_layer_mapping.items()] + \
           [Score(name='edge_ratio', type='anatomy', y=anatomy_score, yerr=0, explanation=None)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_filepath', type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sorted',
                                                             'vgg16-weights_imagenet-activations.pkl')),
                        help='one or more filepaths to the model activations')
    parser.add_argument('--model', type=str, default=None, choices=models.model_mappings.keys(),
                        help='name of the model. Inferred from `--activations_filepath` if None')
    parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'], help='region(s) in brain to compare to')
    parser.add_argument('--map_all_layers', action='store_true', default=True)
    parser.add_argument('--no-map_all_layers', action='store_false', dest='map_all_layers')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running with args {}".format(vars(args)))

    scores = score_model_activations(args.activations_filepath, regions=args.regions, model_name=args.model,
                                     map_all_layers=args.map_all_layers)
    print(scores)


if __name__ == '__main__':
    main()
