import logging

import numpy as np
from numpy.random.mtrand import RandomState
from tqdm import tqdm
from xarray import DataArray

from brainscore.metrics import Score
from brainscore.metrics.ceiling import SpearmanBrownCorrection
from brainscore.metrics.regression import pls_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, apply_aggregate, CrossValidation, \
    subset, standard_error_of_the_mean
from neural_nlp import models
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store_xarray, store

_logger = logging.getLogger(__name__)


def run(model, layers=None):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers)


@store_xarray(identifier_ignore=['layers', 'prerun'], combine_fields={'layers': 'layer'})
def _run(model, layers, prerun=True):
    _logger.info('Running benchmark')
    benchmark = VoxelBenchmark()
    print(benchmark.ceiling)

    if prerun:
        get_activations(model_identifier=model, layers=layers, stimuli=benchmark._target_assembly.stimulus_set)

    layer_scores = []
    for layer in tqdm(layers, desc='layers'):
        candidate = lambda stimuli: get_activations(model_identifier=model, layers=[layer], stimuli=stimuli)
        layer_score = benchmark(candidate)
        layer_score = layer_score.expand_dims('layer')
        layer_score['layer'] = [layer]
        layer_scores.append(layer_score)
    layer_scores = Score.merge(*layer_scores)
    layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    return layer_scores


class VoxelBenchmark:
    def __init__(self):
        _logger.info('Loading neural data')
        neural_data = load_voxels()
        # leave-out Elvis story
        neural_data = neural_data[{'presentation': [story != 'Elvis' for story in neural_data['story'].values]}]
        neural_data.attrs['stimulus_set'] = neural_data.attrs['stimulus_set'][
            [row.story != 'Elvis' for row in neural_data.attrs['stimulus_set'].itertuples()]]
        # note that even though all subjects in this dataset now have seen all stories,
        # there could still be NaN neural data at this point, e.g. from non-collected MD
        neural_data = neural_data.sel(region='language')  # for now
        assert not np.isnan(neural_data).any()
        self._target_assembly = neural_data

        self._regression = pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))
        self._correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        self._metric = CrossRegressedCorrelation(
            regression=self._regression, correlation=self._correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='story'))
        self._cross_subject = CartesianProduct(dividers=['subject_UID'])

    @property
    @store()
    def ceiling(self):
        cross_validation = CrossValidation(split_coord='stimulus_id', stratification_coord='story', splits=2)

        def ceiling_apply(train_source, train_target, test_source, test_target):
            self._regression.fit(train_source, train_target)
            prediction = self._regression.predict(test_source)
            score = self._correlation(prediction, test_target)
            return score

        subjects = list(sorted(set(self._target_assembly['subject_UID'].values)))
        split_scores = []
        for heldout_subject in tqdm(subjects, desc='subject holdout'):
            subject_pool = list(sorted(set(subjects) - {heldout_subject}))
            indexer_pool = DataArray(np.zeros(len(subject_pool)), coords={'subject_UID': subject_pool},
                                     dims=['subject_UID']).stack(neuroid=['subject_UID'])
            heldout_indexer = DataArray(np.zeros(1), coords={'subject_UID': [heldout_subject]},
                                        dims=['subject_UID']).stack(neuroid=['subject_UID'])
            subject_pool = subset(self._target_assembly, indexer_pool, dims_must_match=False)
            heldout = subset(self._target_assembly, heldout_indexer, dims_must_match=False)
            split_score = cross_validation(subject_pool, heldout, apply=ceiling_apply, aggregate=self._metric.aggregate)
            split_score = split_score.expand_dims('heldout_subject')
            split_score['heldout_subject'] = [heldout_subject]
            split_score.attrs[Score.RAW_VALUES_KEY] = split_score.attrs[Score.RAW_VALUES_KEY]
            split_scores.append(split_score)
        consistency = Score.merge(*split_scores)
        error = standard_error_of_the_mean(consistency.sel(aggregation='center'), 'heldout_subject')
        consistency = apply_aggregate(lambda scores: scores.mean('heldout_subject'), consistency)
        consistency.loc[{'aggregation': 'error'}] = error
        return consistency

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

        subject_scores = self._cross_subject(self._target_assembly, apply=
        lambda subject_assembly: self._apply_subject(model_activations, subject_assembly))
        score = apply_aggregate(lambda scores: scores.median('subject_UID'), subject_scores)
        return score

    def _apply_subject(self, source_assembly, subject_assembly):
        assert not np.isnan(subject_assembly).any()
        return self._metric(source_assembly, subject_assembly)
