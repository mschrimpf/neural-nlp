import argparse
import logging
import sys

from neural_nlp import models, stimuli, run

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=models.model_mappings.keys())
parser.add_argument('--model_weights', type=str, default=models._Defaults.model_weights)
parser.add_argument('--no-model_weights', action='store_const', const=None, dest='model_weights')
parser.add_argument('--dataset', type=str, required=True, choices=stimuli.mappings.keys())
parser.add_argument('--log_level', type=str, default='INFO')
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
_logger.info("Running with args %s", vars(args))

run(model=args.model, model_weights=args.model_weights, dataset_name=args.dataset)
