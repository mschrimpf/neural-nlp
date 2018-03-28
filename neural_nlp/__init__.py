import logging

from mkgu.metrics import Benchmark
from mkgu.metrics.rdm import RDMMetric

from neural_nlp import models
from neural_nlp.models import ActivationsWorker
from neural_nlp.neural_data import load as load_neural_data

logger = logging.getLogger(__name__)


class _Defaults(object):
    regions = ('V4', 'IT')


def run(model, dataset_name, model_weights=models._Defaults.model_weights):
    logger.info('Computing activations')
    activations_worker = ActivationsWorker(model_name=model, model_weights=model_weights)
    activations = activations_worker(dataset_name=dataset_name)

    logger.info('Computing scores')
    neural_data = load_neural_data()
    benchmark = Benchmark(metric=RDMMetric(), target_data=neural_data)
    scores = benchmark(activations)

    logger.info('Results')
    print(scores)

    return scores
