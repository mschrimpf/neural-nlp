import logging

from mkgu.metrics.rdm import RDMCorrelationCoefficient, RDM

from neural_nlp import models
from neural_nlp.models import get_activations
from neural_nlp.neural_data import load_rdm_sentences as load_neural_rdms
from neural_nlp.utils import store

_logger = logging.getLogger(__name__)


@store()
def run(model, stimulus_set):
    _logger.info('Computing activations')
    model_activations = get_activations(model_name=model, stimulus_set_name=stimulus_set)
    model_activations = RDM()(model_activations)

    _logger.info('Loading neural data')
    neural_data = load_neural_rdms()
    neural_data = neural_data.mean(dim='subject')
    neural_data = neural_data.rename({'sentence': 'stimulus'})

    _logger.info('Computing scores')
    similarity = RDMCorrelationCoefficient()
    scores = similarity(model_activations, neural_data, rdm_dim='stimulus')
    return scores
