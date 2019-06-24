import logging

import numpy as np
from brainio_base.assemblies import DataAssembly, walk_coords
from tqdm import tqdm
from xarray import DataArray

from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity
from brainscore.metrics.regression import pls_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, subset, standard_error_of_the_mean, \
    apply_aggregate
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences
from neural_nlp.stimuli import load_stimuli
from result_caching import store

_logger = logging.getLogger(__name__)


class VoxelBenchmark:
    def __init__(self):
        _logger.info('Loading neural data')
        assembly = load_voxels()
        # leave-out Elvis story
        assembly = assembly[{'presentation': [story != 'Elvis' for story in assembly['story'].values]}]
        assembly.attrs['stimulus_set'] = assembly.attrs['stimulus_set'][
            [row.story != 'Elvis' for row in assembly.attrs['stimulus_set'].itertuples()]]
        # note that even though all subjects in this dataset now have seen all stories,
        # there could still be NaN neural data at this point, e.g. from non-collected MD
        assembly = assembly.sel(region='language')  # for now
        assert not np.isnan(assembly).any()
        self._target_assembly = assembly

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

        _logger.info('Scoring across subjects')
        subject_scores = self._cross_subject(self._target_assembly, apply=
        lambda subject_assembly: self._apply_subject(model_activations, subject_assembly))
        score = apply_aggregate(lambda scores: scores.median('subject_UID'), subject_scores)
        return score

    def _apply_subject(self, source_assembly, subject_assembly):
        assert not np.isnan(subject_assembly).any()
        return self._metric(source_assembly, subject_assembly)


class RDMBenchmark:
    def __init__(self):
        assemblies = self._load_rdms()
        assemblies = {story: rdm for story, rdm in assemblies.items() if story != 'Elvis'}
        self._target_assemblies = assemblies
        self._metric = RDMSimilarityCrossValidated()
        self._cross_region = CartesianProduct(dividers=['region'])

    def _load_rdms(self, roi_filter='from90to100', bold_shift_seconds=4):
        assemblies = {}
        for story in ['Boar', 'KingOfBirds', 'Elvis', 'HighSchool', 'MatchstickSeller']:
            assembly = load_rdm_sentences(story=story, roi_filter=roi_filter, bold_shift_seconds=bold_shift_seconds)
            assembly = assembly.mean(dim='subject')
            stimulus_set = load_stimuli(f'naturalistic-neural-reduced.{story}')
            assembly.attrs['stimulus_set'] = stimulus_set
            assemblies[story] = assembly
        return assemblies

    def __call__(self, candidate):
        scores = []
        for story, story_assembly in self._target_assemblies.items():
            source_assembly = candidate(stimuli=story_assembly.stimulus_set)
            score = self._cross_region(story_assembly,
                                       apply=lambda region_assembly: self._metric(source_assembly, region_assembly))
            score = score.expand_dims('story')
            score['story'] = [story]
            scores.append(score)
        score = Score.merge(*scores)
        return score


class RDMSimilarityCrossValidated:
    # adapted from
    # https://github.com/brain-score/brain-score/blob/3d59d7a841fca63a5d346e599143f547560b5082/brainscore/metrics/rdm.py#L8

    class LeaveOneOutWrapper:
        def __init__(self, metric):
            self._metric = metric

        def __call__(self, train_source, train_target, test_source, test_target):
            # compare assemblies for a single split. we ignore the 10% train ("leave-one-out") and only use test.
            score = self._metric(test_source, test_target)
            return DataAssembly(score)

    def __init__(self, stimulus_coord='stimulus_sentence'):
        self._rdm = RDM()
        self._similarity = RDMSimilarity(comparison_coord=stimulus_coord)
        self._cross_validation = CrossValidation(test_size=.9,  # leave 10% out
                                                 split_coord=stimulus_coord, stratification_coord=None)

    def __call__(self, model_activations, target_rdm):
        model_activations = align(model_activations, target_rdm, on='stimulus_sentence')
        model_rdm = self._rdm(model_activations)
        leave_one_out = self.LeaveOneOutWrapper(self._similarity)
        # multi-dimensional coords with repeated dimensions not yet supported in CrossValidation
        drop_coords = [coord for coord, dims, value in walk_coords(target_rdm) if dims == ('stimulus', 'stimulus')]
        target_rdm = target_rdm.drop(drop_coords)
        return self._cross_validation(model_rdm, target_rdm, apply=leave_one_out)


def align(source, target, on):
    source_values, target_values = source[on].values.tolist(), target[on].values
    indices = [source_values.index(value) for value in target_values]
    assert len(source[on].dims) == 1, "multi-dimensional coordinates not implemented"
    dim = source[on].dims[0]
    dim_indices = {_dim: slice(None) if _dim != dim else indices for _dim in source.dims}
    aligned = source.isel(**dim_indices)
    return aligned


class fROIBenchmark:
    def __init__(self):
        assembly = load_voxels()
        # leave-out Elvis story
        assembly = assembly[{'presentation': [story != 'Elvis' for story in assembly['story'].values]}]
        assembly.attrs['stimulus_set'] = assembly.attrs['stimulus_set'][
            [row.story != 'Elvis' for row in assembly.attrs['stimulus_set'].itertuples()]]
        assembly = self.average_subregions(assembly)
        self._target_assembly = assembly

        self._regression = pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))
        self._correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        self._metric = CrossRegressedCorrelation(
            regression=self._regression, correlation=self._correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='story'))

    def average_subregions(self, assembly):
        del assembly['threshold']
        assembly = assembly.multi_dim_apply(['stimulus_id', 'fROI_area'], lambda group, **_: group.mean())
        _, index = np.unique(assembly['fROI_area'], return_index=True)
        assembly = assembly.isel(neuroid=index)
        return assembly

    def __call__(self, candidate):
        _logger.info('Computing activations')
        model_activations = candidate(stimuli=self._target_assembly.attrs['stimulus_set'])
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        return self._metric(model_activations, self._target_assembly)


class VoxelPCABenchmark:
    def __init__(self):
        pass

    def __call__(self, candidate):
        pass
