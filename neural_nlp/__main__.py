import argparse
import logging
import sys

from neural_nlp import models, stimuli, run

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=models._model_mappings.keys())
parser.add_argument('--stimulus_set', type=str, required=True, choices=stimuli._mappings.keys())
parser.add_argument('--log_level', type=str, default='INFO')
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(args.log_level))
_logger.info("Running with args %s", vars(args))

run(model=args.model, stimulus_set=args.stimulus_set)
