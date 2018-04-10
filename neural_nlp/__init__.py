import logging

import numpy as np
import xarray as xr
from mkgu.metrics.rdm import RDMCorrelationCoefficient

from neural_nlp import models
from neural_nlp.models import ActivationsWorker
from neural_nlp.neural_data import load_rdms as load_neural_data

_logger = logging.getLogger(__name__)


class _Defaults(object):
    regions = ('V4', 'IT')


def run(model, dataset_name, model_weights=models._Defaults.model_weights):
    _logger.info('Computing activations')
    activations_worker = ActivationsWorker(model_name=model, model_weights=model_weights)
    activations = activations_worker(dataset_name=dataset_name)

    _logger.info('Loading neural data')
    neural_data = load_neural_data()
    # xarray MultiIndexes everything otherwise
    del neural_data['story']
    del neural_data['roi_low']
    del neural_data['roi_high']
    neural_data = neural_data.mean(dim='subject')

    _logger.info('Running spotlight search')
    run_spotlight_search(activations, neural_data)


def run_spotlight_search(model_activations, target_activations):
    """
    Use size of the source activations to determine the size of the spotlight
    and shift over the target activations
    """
    similarity = RDMCorrelationCoefficient()
    spotlight_size = len(model_activations['stimulus'])
    scores = []
    for spotlight_start in range(len(target_activations['timepoint']) - spotlight_size):
        spotlight_end = spotlight_start + spotlight_size
        target_spotlight = target_activations.sel(timepoint=list(range(spotlight_start, spotlight_end)))
        # mock timepoint as a stimulus
        target_spotlight = target_spotlight.rename({'timepoint': 'stimulus'})
        score = similarity(model_activations, target_spotlight, rdm_dim='stimulus')
        spotlight_coords = {'spotlight_start': [spotlight_start],
                            'spotlight_end': ('spotlight_start', [spotlight_end]),
                            'spotlight_size': ('spotlight_start', [spotlight_size])}
        spotlight = xr.DataArray(np.ones(1), coords=spotlight_coords, dims='spotlight_start')
        score = score * spotlight
        _logger.info("Spotlight {} - {}: {}".format(spotlight_start, spotlight_end, score.values))
        scores.append(score)
    return xr.concat(scores, 'spotlight_start')
