import argparse
import logging
import sys

from neural_metrics import models, run

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--model', type=str, required=True, choices=models.model_mappings.keys())
parser.add_argument('--model_weights', type=str, default=models._Defaults.model_weights)
parser.add_argument('--layers', type=str, nargs='+', required=True)
parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'])
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
logger.info("Running with args %s", vars(args))

run(model=args.model, model_weights=args.model_weights, layers=args.layers, regions=args.regions, save_plot=True)
