import argparse
import logging
import os
import sys

from neural_metrics.compare import metrics_for_activations
from neural_metrics.models import activations_for_model, model_mappings
from neural_metrics.plot import plot_layer_correlations

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
parser.add_argument('--model', type=str, required=True, choices=model_mappings.keys())
parser.add_argument('--layers', type=str, nargs='+', required=True)
parser.add_argument('--regions', type=str, nargs='+', default=['V4', 'IT'])
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
logger.info("Running with args %s", vars(args))

logger.info('Computing activations')
activations_savepath = activations_for_model(model=args.model, layers=args.layers, use_cached=True)
logger.info('Computing metrics')
metrics_savepaths = [metrics_for_activations(activations_savepath, region=region, use_cached=True)
                     for region in args.regions]

logger.info('Plotting')
file_name = os.path.splitext(os.path.basename(activations_savepath))[0]
output_filepath = os.path.join(os.path.dirname(__file__), '..', 'results',
                               '{}-regions_{}{}'.format(file_name, ''.join(args.regions), '.svg'))
plot_layer_correlations(metrics_savepaths, labels=args.regions, output_filepath=output_filepath)
