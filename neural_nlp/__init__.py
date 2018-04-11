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

    _logger.info('Running searchlight search')
    scores = run_searchlight_search(activations, neural_data)
    searchlights = scores.argmax(dim='searchlight_start')
    scores = scores.max(dim='searchlight_start')
    scores['searchlight_start_max'] = 'region', searchlights
    return scores


def run_searchlight_search(model_rdms, target_rdms):
    """
    Use size of the model RDMs to determine the size of the searchlight
    and shift over the target RDMs
    """
    similarity = RDMCorrelationCoefficient()
    searchlight_size = len(model_rdms['stimulus'])
    scores = []
    for searchlight_start in range(len(target_rdms['timepoint']) - searchlight_size):
        searchlight_end = searchlight_start + searchlight_size
        target_searchlight = target_rdms.sel(timepoint=list(range(searchlight_start, searchlight_end)))
        # mock timepoint as a stimulus
        target_searchlight = target_searchlight.rename({'timepoint': 'stimulus'})
        score = similarity(model_rdms, target_searchlight, rdm_dim='stimulus')
        searchlight_coords = {'searchlight_start': [searchlight_start],
                              'searchlight_end': ('searchlight_start', [searchlight_end]),
                              'searchlight_size': ('searchlight_start', [searchlight_size])}
        searchlight = xr.DataArray(np.ones(1), coords=searchlight_coords, dims='searchlight_start')
        score = score * searchlight
        _logger.info("searchlight {} - {}: {}".format(searchlight_start, searchlight_end, score.values))
        scores.append(score)
    return xr.concat(scores, 'searchlight_start')
