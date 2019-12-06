import logging
import warnings

import itertools
import numpy as np
from brainio_base.assemblies import DataAssembly, walk_coords, merge_data_arrays, NeuroidAssembly
from tqdm import tqdm
from sklearn.linear_model import Ridge
from xarray import DataArray

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity, RDMCrossValidated
from brainscore.metrics.regression import pls_regression, linear_regression, pearsonr_correlation, \
    CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, subset, standard_error_of_the_mean, \
    apply_aggregate
from brainscore.metrics.xarray_utils import XarrayRegression
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences, load_Pereira2018
from neural_nlp.neural_data.ecog_greta import load_Fedorenko2016
from neural_nlp.stimuli import load_stimuli, StimulusSet
from neural_nlp.utils import ordered_set
from result_caching import store

_logger = logging.getLogger(__name__)


def cg_regression(xarray_kwargs=None):
    regression = Ridge()  # solver='sparse_cg')
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


class Invert:
    def __init__(self, metric):
        self._metric = metric

    def __call__(self, source, target):
        source, target = target, source
        return self._metric(source, target)


class StoriesTransformerWordmeanBenchmark:
    def __init__(self, bold_shift=None):
        from neural_nlp import get_activations
        self._target_assemblies = {story: get_activations(
            model_identifier='transformer-wordmean', layers=[f'encoder.transformer.0.feed_forward.dropout_2'],
            stimuli=load_stimuli(f'naturalistic-neural-reduced.{story}'))
            for story in ['Boar', 'KingOfBirds', 'Elvis', 'HighSchool', 'MatchstickSeller']}
        self._metric = RDMSimilarityCrossValidated()

    def __call__(self, candidate):
        scores = []
        for story, story_assembly in self._target_assemblies.items():
            stimulus_set = load_stimuli(f'naturalistic-neural-reduced.{story}')
            source_assembly = candidate(stimuli=stimulus_set)
            story_assembly_rdm = self._metric._rdm(story_assembly)
            score = self._metric(source_assembly, story_assembly_rdm)
            score = score.expand_dims('story')
            score['story'] = [story]
            scores.append(score)
        score = Score.merge(*scores)
        score = apply_aggregate(lambda score: score.mean('story'), score)
        return score


class _StoriesVoxelBenchmark(Benchmark):
    def __init__(self, regression, correlation, metric, bold_shift=4):
        _logger.info('Loading neural data')
        assembly = load_voxels(bold_shift_seconds=bold_shift)
        # leave-out Elvis story
        assembly = assembly[{'presentation': [story != 'Elvis' for story in assembly['story'].values]}]
        assembly.attrs['stimulus_set'] = assembly.attrs['stimulus_set'][
            [row.story != 'Elvis' for row in assembly.attrs['stimulus_set'].itertuples()]]
        # note that even though all subjects in this dataset now have seen all stories,
        # there could still be NaN neural data at this point, e.g. from non-collected MD
        assembly = assembly.sel(region='language')  # for now
        assert not np.isnan(assembly).any()
        self._target_assembly = assembly

        self._regression = regression
        self._correlation = correlation
        self._metric = metric
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


class StoriesVoxelEncoding(_StoriesVoxelBenchmark):
    def __init__(self, bold_shift=4):
        regression = pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='story'))
        super(StoriesVoxelEncoding, self).__init__(bold_shift=bold_shift,
                                                   regression=regression, correlation=correlation, metric=metric)


class StoriesVoxelEncodingCG(_StoriesVoxelBenchmark):
    def __init__(self, bold_shift=4):
        regression = cg_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='story'))
        super(StoriesVoxelEncodingCG, self).__init__(bold_shift=bold_shift,
                                                     regression=regression, correlation=correlation, metric=metric)


class StoriesVoxelDecoding(_StoriesVoxelBenchmark):
    def __init__(self, bold_shift=4):
        regression = pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='story'))
        metric = Invert(metric)
        super(StoriesVoxelDecoding, self).__init__(bold_shift=bold_shift,
                                                   regression=regression, correlation=correlation, metric=metric)


class StoriesRDMBenchmark:
    def __init__(self, bold_shift=4):
        assemblies = self._load_rdms(bold_shift_seconds=bold_shift)
        assemblies = {story: rdm for story, rdm in assemblies.items() if story != 'Elvis'}
        self._target_assemblies = assemblies
        self._metric = RDMSimilarityCrossValidated()
        self._cross_region = CartesianProduct(dividers=['region'])

    def _load_rdms(self, roi_filter='from90to100', bold_shift_seconds=4):
        assemblies = {}
        for story in ['Boar', 'KingOfBirds', 'Elvis', 'HighSchool', 'MatchstickSeller']:
            assembly = load_rdm_sentences(story=story, roi_filter=roi_filter, bold_shift_seconds=bold_shift_seconds)
            assembly = assembly.mean(dim='subject')
            stimulus_set_identifier = f'naturalistic-neural-reduced.{story}'
            stimulus_set = load_stimuli(stimulus_set_identifier)
            stimulus_set = StimulusSet({'sentence': stimulus_set})
            stimulus_set.name = stimulus_set_identifier
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
        score = apply_aggregate(lambda score: score.mean('story'), score)
        score = apply_aggregate(lambda score: score.mean('region'), score)
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
        values = model_rdm.values
        if np.isnan(values.flatten()).any():
            warnings.warn(f"{np.isnan(values.flatten()).sum()} nan values found in model rdm - setting to 0")
            values[np.isnan(values)] = 0
            model_rdm = type(model_rdm)(values, coords={coord: (dims, vals) for coord, dims, vals in
                                                        walk_coords(model_rdm)}, dims=model_rdm.dims)
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


class StoriesfROIBenchmark:
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


class _PereiraBenchmark(Benchmark):
    def __init__(self, metric):
        self._target_assembly = load_Pereira2018()
        self._metric = metric
        self._cross_subject = CartesianProduct(dividers=['subject'])

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to_stories(candidate, stimulus_set)
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring across subjects')
        subject_scores = self._cross_subject(self._target_assembly, apply=
        lambda subject_assembly: self._apply_subject(model_activations, subject_assembly))
        score = apply_aggregate(lambda scores: scores.median('subject'), subject_scores)
        return score

    def _apply_subject(self, source_assembly, subject_assembly):
        subject_assembly = subject_assembly.dropna('presentation')  # some subjects have only done one experiment
        assert len(subject_assembly['presentation']) in [243, 384, 243 + 384]
        assert not np.isnan(subject_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in subject_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._metric(source_assembly, subject_assembly)

    def ceiling(self):
        return None


def listen_to_stories(candidate, stimulus_set):
    activations = []
    for story in ordered_set(stimulus_set['story'].values):
        story_stimuli = stimulus_set[stimulus_set['story'] == story]
        story_stimuli.name = f"{stimulus_set.name}-{story}"
        story_activations = candidate(stimuli=story_stimuli)
        activations.append(story_activations)
    model_activations = merge_data_arrays(activations)
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    return model_activations


class PereiraEncoding(_PereiraBenchmark):
    def __init__(self, bold_shift=None):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=3))
        super(PereiraEncoding, self).__init__(metric=metric)


class PereiraEncodingMin(_PereiraBenchmark):
    def __init__(self, bold_shift=None):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=2))
        super(PereiraEncodingMin, self).__init__(metric=metric)
        self._target_assembly = self._target_assembly.sel(subject='M02')
        self._target_assembly['subject'] = 'neuroid', ['M02'] * len(self._target_assembly['neuroid'])
        self._target_assembly = NeuroidAssembly(self._target_assembly)  # re-index with subject


class PereiraEncodingCG(_PereiraBenchmark):
    def __init__(self, bold_shift=None):
        metric = CrossRegressedCorrelation(
            regression=cg_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=3))
        super(PereiraEncodingCG, self).__init__(metric=metric)


class PereiraDecoding(_PereiraBenchmark):
    def __init__(self, bold_shift=None):
        metric = CrossRegressedCorrelation(
            regression=pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None))
        metric = Invert(metric)
        super(PereiraDecoding, self).__init__(metric=metric)


class PereiraRDM(_PereiraBenchmark):
    def __init__(self, bold_shift=None):
        metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=3))
        super(PereiraRDM, self).__init__(metric=metric)

def read_words(candidate, stimulus_set): # This is a new version of the listen_to_stories function
    # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
    activations = []
    for sentence in ordered_set(stimulus_set['sentence_id'].values):
        sentence_stimuli = stimulus_set[stimulus_set['sentence_id'] == sentence]
        sentence_stimuli.name = f"{stimulus_set.name}-{sentence}"
        sentence_activations = candidate(stimuli=sentence_stimuli)
        activations.append(sentence_activations)
    model_activations = merge_data_arrays(activations)
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    
    return model_activations
        
        
class FedorenkoBenchmark:
    def __init__(self, bold_shift=None):
        assembly = load_Fedorenko2016() 
        self._target_assembly = assembly
        
        self._regression = pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')) # word
        self._correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        self._metric = CrossRegressedCorrelation(
            regression=self._regression, correlation=self._correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='stimulus_id'))

# Remove the ceiling
    # @property
    # @store()
    # def ceiling(self):
    #     cross_validation = CrossValidation(split_coord='stimulus_id', stratification_coord='stimulus_id', splits=2) # still assuming a stratification cord along the word/timepoint dimension
    #
    #     def ceiling_apply(train_source, train_target, test_source, test_target):
    #         self._regression.fit(train_source, train_target)
    #         prediction = self._regression.predict(test_source)
    #         score = self._correlation(prediction, test_target)
    #         return score
    #
    #     subjects = list(sorted(set(self._target_assembly['subject_UID'].values))) # the subjectUID from the ECoG assembly packaging
    #     split_scores = []
    #
    #     for heldout_subject in tqdm(subjects, desc='subject holdout'):
    #         subject_pool = list(sorted(set(subjects) - {heldout_subject}))
    #         indexer_pool = DataArray(np.zeros(len(subject_pool)), coords={'subject_UID': subject_pool},
    #                                  dims=['subject_UID']).stack(neuroid=['subject_UID'])
    #         heldout_indexer = DataArray(np.zeros(1), coords={'subject_UID': [heldout_subject]},
    #                                     dims=['subject_UID']).stack(neuroid=['subject_UID'])
    #         subject_pool = subset(self._target_assembly, indexer_pool, dims_must_match=False)
    #         heldout = subset(self._target_assembly, heldout_indexer, dims_must_match=False)
    #         split_score = cross_validation(subject_pool, heldout, apply=ceiling_apply, aggregate=self._metric.aggregate)
    #         split_score = split_score.expand_dims('heldout_subject')
    #         split_score['heldout_subject'] = [heldout_subject]
    #         split_score.attrs[Score.RAW_VALUES_KEY] = split_score.attrs[Score.RAW_VALUES_KEY]
    #         split_scores.append(split_score)
    #     consistency = Score.merge(*split_scores)
    #     error = standard_error_of_the_mean(consistency.sel(aggregation='center'), 'heldout_subject')
    #     consistency = apply_aggregate(lambda scores: scores.mean('heldout_subject'), consistency)
    #     consistency.loc[{'aggregation': 'error'}] = error
    #     return consistency
    def ceiling(self):
        return None
        
    def __call__(self, candidate):
        _logger.info('Computing activations')
        
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        
        model_activations = read_words(candidate, stimulus_set)
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        return self._metric(model_activations, self._target_assembly)

    
    
class FedorenkoBenchmarkMean:
    def __init__(self, bold_shift=None):
        assembly = load_Fedorenko2016() 
        self._target_assembly = assembly
                
        # avg code 
        # packaging file, change the benchmark to include averaging. 
        
        self._regression = pls_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')) # word
        self._correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
        self._metric = CrossRegressedCorrelation(
            regression=self._regression, correlation=self._correlation,
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='stimulus_id'))
    
    def __call__(self, candidate):
        
        _logger.info('Computing activations')
        
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        
        model_activations = read_words(candidate, stimulus_set)
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        
        # average here, can I simply hard code the averaging across sentences?
        # model_activations = 
        
        print('Model activations:   ', model_activations)
        
        return self._metric(model_activations, self._target_assembly)
        
        
#         scores = []
#         for story, story_assembly in self._target_assemblies.items():
#             stimulus_set = load_stimuli(f'naturalistic-neural-reduced.{story}')
#             source_assembly = candidate(stimuli=stimulus_set)
#             story_assembly_rdm = self._metric._rdm(story_assembly)
#             score = self._metric(source_assembly, story_assembly_rdm)
#             score = score.expand_dims('story')
#             score['story'] = [story]
#             scores.append(score)
#         score = Score.merge(*scores)
#         score = apply_aggregate(lambda score: score.mean('story'), score)
#         return score

    
    def ceiling(self):
        return None
    
    
    
    

benchmark_pool = {
    'voxel-encoding': StoriesVoxelEncoding,
    'stories-voxel-encoding-cg': StoriesVoxelEncodingCG,
    'voxel-decoding': StoriesVoxelDecoding,
    'fROI': StoriesfROIBenchmark,
    'rdm': StoriesRDMBenchmark,
    'Pereira2018-encoding': PereiraEncoding,
    'Pereira2018-encoding-min': PereiraEncodingMin,
    'Pereira2018-encoding-cg': PereiraEncodingCG,
    'Pereira2018-decoding': PereiraDecoding,
    'Pereira2018-rdm': PereiraRDM,
    'transformer-wordmean': StoriesTransformerWordmeanBenchmark,
    'Fedorenko2016': FedorenkoBenchmark,
    'Fedorenko2016Mean': FedorenkoBenchmarkMean,
}

# Added dec 5th
# if __name__ == '__main__':
#     FedorenkoBenchmark()