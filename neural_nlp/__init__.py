import logging
import os

import numpy as np
import xarray as xr
from mkgu.metrics.rdm import RDMCorrelationCoefficient, RDM

from neural_nlp import models
from neural_nlp.models import ActivationsWorker
from neural_nlp.neural_data import load_rdms as load_neural_rdms
from neural_nlp.utils import store

_logger = logging.getLogger(__name__)


class _Defaults(object):
    regions = ('V4', 'IT')


@store(storage_directory=os.path.join(os.path.dirname(__file__), '..', 'output'))
def run(model, dataset_name, model_weights=models._Defaults.model_weights):
    _logger.info('Computing activations')
    activations_worker = ActivationsWorker(model_name=model, model_weights=model_weights)
    activations = activations_worker(dataset_name=dataset_name)
    activations = RDM()(activations)

    _logger.info('Loading neural data')
    neural_data = load_neural_rdms()
    # xarray MultiIndexes everything otherwise
    del neural_data['story']
    del neural_data['roi_low']
    del neural_data['roi_high']
    neural_data = neural_data.mean(dim='subject')

    _logger.info('Running spotlight search')
    scores = run_spotlight_search(activations, neural_data)
    spotlights = scores.argmax(dim='spotlight_start')
    scores = scores.max(dim='spotlight_start')
    scores['spotlight_start_max'] = 'region', spotlights
    return scores


def run_spotlight_search(model_rdms, target_rdms):
    """
    Use size of the model RDMs to determine the size of the spotlight
    and shift over the target RDMs
    """
    similarity = RDMCorrelationCoefficient()
    spotlight_size = len(model_rdms['stimulus'])
    scores = []
    for spotlight_start in range(len(target_rdms['timepoint']) - spotlight_size):
        spotlight_end = spotlight_start + spotlight_size
        target_spotlight = target_rdms.sel(timepoint=list(range(spotlight_start, spotlight_end)))
        # mock timepoint as a stimulus
        target_spotlight = target_spotlight.rename({'timepoint': 'stimulus'})
        score = similarity(model_rdms, target_spotlight, rdm_dim='stimulus')
        spotlight_coords = {'spotlight_start': [spotlight_start],
                            'spotlight_end': ('spotlight_start', [spotlight_end]),
                            'spotlight_size': ('spotlight_start', [spotlight_size])}
        spotlight = xr.DataArray(np.ones(1), coords=spotlight_coords, dims='spotlight_start')
        score = score * spotlight
        _logger.info("Spotlight {} - {}: {}".format(spotlight_start, spotlight_end, score.values))
        scores.append(score)
    return xr.concat(scores, 'spotlight_start')
