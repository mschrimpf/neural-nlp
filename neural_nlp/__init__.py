import logging

import numpy as np

from brainscore.metrics.regression import pls_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, apply_aggregate
from neural_nlp import models
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store_xarray

_logger = logging.getLogger(__name__)


def run(model, layers=None):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers)


@store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
def _run(model, layers):
    def candidate(stimuli):
        return get_activations(model_identifier=model, layers=layers, stimuli=stimuli)

    _logger.info('Running benchmark')
    benchmark = NaturalisticStoriesBenchmark()
    scores = benchmark(candidate)
    return scores


class NaturalisticStoriesBenchmark:
    def __init__(self):
        _logger.info('Loading neural data')
        neural_data = load_voxels()
        # leave-out ELvis story
        neural_data = neural_data[{'presentation': [story != 'Elvis' for story in neural_data['story'].values]}]
        neural_data.attrs['stimulus_set'] = neural_data.attrs['stimulus_set'][
            [row.story != 'Elvis' for row in neural_data.attrs['stimulus_set'].itertuples()]]
        # exclude subjects that have not seen all stories
        subjects = np.isnan(neural_data).any('presentation').groupby('subject').apply(
            lambda subject_neuroids: subject_neuroids.any())
        subjects = subjects['subject'].values[~subjects.values]
        neural_data = neural_data[{'neuroid': [subject in subjects for subject in neural_data['subject'].values]}]
        assert not np.isnan(neural_data).any()
        self._target_assembly = neural_data

        self._metric = CrossRegressedCorrelation(
            regression=pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='story'))
        self._cross_subject = CartesianProduct(dividers=['subject'])

    def __call__(self, candidate):
        _logger.info('Computing activations')
        model_activations = candidate(stimuli=self._target_assembly.attrs['stimulus_set'])
        # since we're presenting all stimuli, including the inter-recording ones, we need to filter
        model_activations = model_activations[
            {'presentation': [sentence in self._target_assembly['stimulus_sentence'].values
                              for sentence in
                              model_activations['stimulus_sentence'].values]}]
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring layers')
        cross_layer = CartesianProduct(dividers=['layer'])
        scores = cross_layer(model_activations, apply=self._apply_layer)
        return scores

    def _apply_layer(self, source_assembly):
        subject_scores = self._cross_subject(self._target_assembly, apply=
        lambda subject_assembly: self._apply_subject(source_assembly, subject_assembly))
        score = apply_aggregate(lambda scores: scores.median('subject'), subject_scores)
        return score

    def _apply_subject(self, source_assembly, subject_assembly):
        return self._metric(source_assembly, subject_assembly)
