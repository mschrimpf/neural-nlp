import argparse
import fire
import logging
import sys
from datetime import datetime

from neural_nlp import score as score_function

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")


def run(benchmark, model, layers=None, subsample=None):
    start = datetime.now()
    score = score_function(model=model, layers=layers, subsample=subsample, benchmark=benchmark)
    end = datetime.now()
    print(score)
    print(f"Duration: {end - start}")


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    fire.Fire(command=FIRE_FLAGS)
