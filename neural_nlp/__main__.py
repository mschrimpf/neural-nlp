import logging
import sys

import argparse
import fire

from neural_nlp import _run
from neural_nlp.models import model_layers

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")


def run(benchmark, model, layers=None, prerun=True, subsample=None):
    layers = layers or model_layers[model]
    score = _run(benchmark=benchmark, model=model, layers=layers, prerun=prerun, subsample=subsample)
    print(score)


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
