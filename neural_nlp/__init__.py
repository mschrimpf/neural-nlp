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
        cross_validation = CrossValidation(split_coord='stimulus_id', stratification_coord='story', splits=1)
        correction = SpearmanBrownCorrection()

        def ceiling_apply(train_source, train_target, test_source, test_target):
            self._regression.fit(train_source, train_target)
            prediction = self._regression.predict(test_source)
            score = self._correlation(prediction, test_target)
            score = correction(score, n=2)
            return score

        subjects = list(set(self._target_assembly['subject_UID'].values))
        rng = RandomState(0)
        split_scores = []
        for subject_split in tqdm(range(10), desc='subject split'):
            rng.shuffle(subjects)
            half1, half2 = subjects[:len(subjects) // 2], subjects[len(subjects) // 2:]
            indexer1 = DataArray(np.zeros(len(half1)), coords={'subject_UID': half1}, dims=['subject_UID']) \
                .stack(neuroid=['subject_UID'])
            indexer2 = DataArray(np.zeros(len(half2)), coords={'subject_UID': half2}, dims=['subject_UID']) \
                .stack(neuroid=['subject_UID'])
            half1 = subset(self._target_assembly, indexer1, dims_must_match=False)
            half2 = subset(self._target_assembly, indexer2, dims_must_match=False)
            split_score = cross_validation(half1, half2, apply=ceiling_apply, aggregate=self._metric.aggregate)
            split_score = split_score.expand_dims('subject_split')
            split_score['subject_split'] = [subject_split]
            split_score.attrs[Score.RAW_VALUES_KEY] = split_score.attrs[Score.RAW_VALUES_KEY].squeeze('split')
            split_scores.append(split_score)
        consistency = Score.merge(*split_scores)
        error = standard_error_of_the_mean(consistency.sel(aggregation='center'), 'subject_split')
        consistency = apply_aggregate(lambda scores: scores.mean('subject_split'), consistency)
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
        cross_layer = CartesianProduct(dividers=['layer'])
        scores = cross_layer(model_activations, apply=self._apply_layer)
        return scores

    def _apply_layer(self, source_assembly):
        subject_scores = self._cross_subject(self._target_assembly, apply=
        lambda subject_assembly: self._apply_subject(source_assembly, subject_assembly))
        score = apply_aggregate(lambda scores: scores.median('subject_UID'), subject_scores)
        return score

    def _apply_subject(self, source_assembly, subject_assembly):
        assert not np.isnan(subject_assembly).any()
        return self._metric(source_assembly, subject_assembly)


