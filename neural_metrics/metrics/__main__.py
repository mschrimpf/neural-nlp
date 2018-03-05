import argparse
import logging
import os
import sys

from neural_metrics import models, score_model_activations

_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations_filepath', type=str, nargs='+',
                        default=[os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'sorted',
                                                              'vgg16-weights_imagenet-activations.pkl'))],
                        help='one or more filepaths to the model activations')
    parser.add_argument('--model', type=str, default=None, choices=models.model_mappings.keys(),
                        help='name of the model. Inferred from `--activations_filepath` if None')
    parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'], help='region(s) in brain to compare to')
    parser.add_argument('--output_directory', type=str, default=None)
    parser.add_argument('--map_all_layers', action='store_true', default=True)
    parser.add_argument('--no-map_all_layers', action='store_false', dest='map_all_layers')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
    _logger.info("Running with args {}".format(vars(args)))

    for activations_filepath in args.activations_filepath:
        print(activations_filepath)
        scores = score_model_activations(activations_filepath, regions=args.regions, model_name=args.model,
                                         map_all_layers=args.map_all_layers, output_directory=args.output_directory)
        print(scores)


if __name__ == '__main__':
    main()
