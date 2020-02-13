import warnings

import itertools
import logging
import numpy as np
from brainio_base.assemblies import DataAssembly, walk_coords, merge_data_arrays, NeuroidAssembly, array_is_element
from tqdm import tqdm, trange

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.rdm import RDM, RDMSimilarity, RDMCrossValidated
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, CrossValidation, standard_error_of_the_mean, \
    apply_aggregate
from brainscore.utils import LazyLoad
from neural_nlp.neural_data.ecog_greta import load_Fedorenko2016
from neural_nlp.neural_data.fmri import load_voxels, load_rdm_sentences, \
    load_Pereira2018_Blank, load_Pereira2018_Blank_languageresiduals
from neural_nlp.stimuli import load_stimuli, StimulusSet
from neural_nlp.utils import ordered_set
from result_caching import store

_logger = logging.getLogger(__name__)


class Invert:
    def __init__(self, metric):
        self._metric = metric

    def __call__(self, source, target):
        source, target = target, source
        return self._metric(source, target)


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


class StoriesVoxelEncoding:
    def __init__(self, bold_shift=4):
        assembly = LazyLoad(lambda: self._load_assembly(bold_shift))
        self._target_assembly = assembly
        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='stimulus_id', neuroid_coord='neuroid_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='stimulus_id', neuroid_coord='neuroid_id'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord='story'))

    def _load_assembly(self, bold_shift):
        assembly = load_voxels(bold_shift_seconds=bold_shift)
        return assembly

    @property
    @store()
    def ceiling(self):
        return holdout_subject_ceiling(assembly=self._target_assembly, subject_column='subject_UID',
                                       metric=self._metric)

    def __call__(self, candidate):
        _logger.info('Computing activations')
        model_activations = listen_to(candidate, self._target_assembly.attrs['stimulus_set'])
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        _logger.info('Scoring model')
        return self._metric(model_activations, self._target_assembly)


class StoriesfROIEncoding(StoriesVoxelEncoding):
    def __init__(self, *args, **kwargs):
        super(StoriesfROIEncoding, self).__init__(*args, **kwargs)

        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='stimulus_id', neuroid_coord='fROI_area'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='stimulus_id', neuroid_coord='fROI_area'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord='story'))

    def _load_assembly(self, bold_shift):
        assembly = super(StoriesfROIEncoding, self)._load_assembly(bold_shift)
        assembly = self.average_subregions(bold_shift=bold_shift, assembly=assembly)
        return assembly

    @store(identifier_ignore=['assembly'])
    def average_subregions(self, bold_shift, assembly):
        attrs = assembly.attrs
        del assembly['threshold']
        # group by stimuli, fROI, subject after one another.
        # this gets rid of adjacent coords unfortunately, but we accept that for now.
        averaged_assembly = assembly.groupby('stimulus_id').apply(
            lambda stimulus_group: stimulus_group.groupby('fROI_area').apply(
                lambda fROI_group: fROI_group.groupby('subject_UID').mean()
            ))
        averaged_assembly = averaged_assembly.stack(presentation=['stimulus_id'], neuroid=['fROI_area', 'subject_UID'])
        # copy presentation coords back since those are needed for e.g. metric stratification
        order = [averaged_assembly['stimulus_id'].values.tolist().index(stimulus_id)
                 for stimulus_id in assembly['stimulus_id'].values]
        for copy_coord, dims, copy_value in walk_coords(assembly):
            if not array_is_element(dims, 'presentation') or hasattr(averaged_assembly, copy_coord):
                continue
            averaged_assembly[copy_coord] = dims, copy_value[order]
        averaged_assembly.attrs = attrs
        return averaged_assembly


class StoriesfROIRDM(StoriesfROIEncoding):
    def __init__(self, *args, **kwargs):
        super(StoriesfROIRDM, self).__init__(*args, **kwargs)
        self._metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))


class _PereiraBenchmark(Benchmark):
    def __init__(self, metric, data_version='base'):
        self._target_assembly = LazyLoad(lambda: self._load_assembly(version=data_version))
        self._single_metric = metric
        self._cross = CartesianProduct(dividers=['experiment', 'atlas'])

    def _metric(self, source_assembly, target_assembly):
        cross_scores = self._cross(target_assembly, apply=
        lambda cross_assembly: self._apply_cross(source_assembly, cross_assembly))
        score = cross_scores.mean(['experiment', 'atlas'])
        return score

    def _load_assembly(self, version='base'):
        assembly = load_Pereira2018_Blank(version=version)
        assembly = assembly.sel(atlas_selection_lower=90)
        assembly = assembly[{'neuroid': [filter_strategy in [np.nan, 'HminusE', 'FIXminusH']
                                         for filter_strategy in assembly['filter_strategy'].values]}]
        return assembly

    def __call__(self, candidate):
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = listen_to(candidate, stimulus_set)
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)

        _logger.info('Scoring across experiments & atlases')
        cross_scores = self._cross(self._target_assembly, apply=
        lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        return cross_scores

    def _apply_cross(self, source_assembly, cross_assembly):
        cross_assembly = cross_assembly.dropna('neuroid')  # some subjects have only done one experiment
        assert len(cross_assembly['presentation']) in [243, 384]
        assert not np.isnan(cross_assembly).any()
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._single_metric(source_assembly, cross_assembly)

    @property
    @store()
    def ceiling(self):
        def compare(pool_candidate, subject_target):
            cross = CartesianProduct(dividers=['atlas'])  # no experiment
            scores = cross(subject_target, apply=
            lambda cross_assembly: self._apply_cross(pool_candidate, cross_assembly))
            return scores

        experiment_scores = []
        for experiment in ordered_set(self._target_assembly['experiment'].values):
            experiment_assembly = self._target_assembly[{'presentation': [
                experiment_value == experiment for experiment_value in self._target_assembly['experiment'].values]}]
            experiment_assembly = experiment_assembly.dropna('neuroid')  # drop subjects that haven't done this one
            scores = holdout_subject_ceiling(experiment_assembly, metric=compare)
            scores = scores.expand_dims('experiment')
            scores['experiment'] = [experiment]
            experiment_scores.append(scores)
        scores = Score.merge(*experiment_scores)
        scores = scores.mean('experiment').mean('atlas')
        return scores


def listen_to(candidate, stimulus_set, reset_column='story', average_sentence=True):
    activations = []
    for story in ordered_set(stimulus_set[reset_column].values):
        story_stimuli = stimulus_set[stimulus_set[reset_column] == story]
        story_stimuli.name = f"{stimulus_set.name}-{story}"
        story_activations = candidate(stimuli=story_stimuli, average_sentence=average_sentence)
        activations.append(story_activations)
    model_activations = merge_data_arrays(activations)
    # merging does not maintain stimulus order. the following orders again
    idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
           itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
    assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
    model_activations = model_activations[{'presentation': idx}]
    return model_activations


class PereiraEncoding(_PereiraBenchmark):
    def __init__(self, **kwargs):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id', stratification_coord=None))
        super(PereiraEncoding, self).__init__(metric=metric, **kwargs)


class PereiraLanguageResidualsEncoding(PereiraEncoding):
    def __init__(self):
        super(PereiraLanguageResidualsEncoding, self).__init__()
        self._target_assembly = LazyLoad(load_Pereira2018_Blank_languageresiduals)


class PereiraICAEncoding(PereiraEncoding):
    def __init__(self):
        super(PereiraICAEncoding, self).__init__(data_version='ICA')


class PereiraDemeanEncoding(PereiraEncoding):
    def __init__(self):
        super(PereiraDemeanEncoding, self).__init__(data_version='Demean')


class PereiraNovisaudEncoding(PereiraEncoding):
    def __init__(self):
        super(PereiraNovisaudEncoding, self).__init__(data_version='NoVisAud')


class PereiraEncodingMin(_PereiraBenchmark):
    def __init__(self):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=2))
        super(PereiraEncodingMin, self).__init__(metric=metric)
        self._target_assembly = self._target_assembly.sel(subject='018')
        self._target_assembly['subject'] = 'neuroid', ['018'] * len(self._target_assembly['neuroid'])
        self._target_assembly = NeuroidAssembly(self._target_assembly)  # re-index with subject


class PereiraDecoding(_PereiraBenchmark):
    def __init__(self):
        metric = CrossRegressedCorrelation(
            regression=linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id')),
            correlation=pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id')),
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None))
        metric = Invert(metric)
        super(PereiraDecoding, self).__init__(metric=metric)


class PereiraRDM(_PereiraBenchmark):
    def __init__(self):
        metric = RDMCrossValidated(
            comparison_coord='stimulus_id',
            crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord=None, splits=5,
                                        kfold=True, test_size=None))
        super(PereiraRDM, self).__init__(metric=metric)


class _Fedorenko2016:
    def __init__(self, identifier, metric):
        self.identifier = identifier
        assembly = load_Fedorenko2016(electrodes='language', version=1)
        self._target_assembly = assembly
        self._metric = metric

    @property
    def ceiling(self):
        return self._ceiling(identifier=self.identifier)

    @store()
    def _ceiling(self, identifier):
        return holdout_subject_ceiling(assembly=self._target_assembly, subject_column='subject_UID',
                                       metric=self._metric)

    def __call__(self, candidate):
        _logger.info('Computing activations')
        stimulus_set = self._target_assembly.attrs['stimulus_set']
        model_activations = self._read_words(candidate, stimulus_set)
        assert (model_activations['stimulus_id'].values == self._target_assembly['stimulus_id'].values).all()
        return self._metric(model_activations, self._target_assembly)

    @classmethod
    def _read_words(cls, candidate, stimulus_set):  # This is a new version of the listen_to_stories function
        # Input: stimulus_set = pandas df, col 1 with sentence ID and 2nd col as word.
        activations = []
        for i, sentence_id in enumerate(ordered_set(stimulus_set['sentence_id'].values)):
            sentence_stimuli = stimulus_set[stimulus_set['sentence_id'] == sentence_id]
            sentence_stimuli = StimulusSet({'sentence': ' '.join(sentence_stimuli['word']),
                                            'sentence_id': list(set(sentence_stimuli['sentence_id']))})
            sentence_stimuli.name = f"{stimulus_set.name}-{sentence_id}"
            sentence_activations = candidate(stimuli=sentence_stimuli, average_sentence=False)
            sentence_activations['stimulus_id'] = ('presentation', 8 * i + np.arange(0, 8))
            sentence_activations['sentence_id'] = ('presentation', [sentence_id] * 8)
            activations.append(sentence_activations)
        model_activations = merge_data_arrays(activations)
        # merging does not maintain stimulus order. the following orders again
        idx = [model_activations['stimulus_id'].values.tolist().index(stimulus_id) for stimulus_id in
               itertools.chain.from_iterable(s['stimulus_id'].values for s in activations)]
        assert len(set(idx)) == len(idx), "Found duplicate indices to order activations"
        model_activations = model_activations[{'presentation': idx}]

        return model_activations

# The original Fedorenko2016 benchmark (version 1) 
def Fedorenko2016EncodingV1():
    regression = linear_regression(xarray_kwargs=dict(stimulus_coord='stimulus_id'))  # word
    correlation = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id'))
    metric = CrossRegressedCorrelation(regression=regression, correlation=correlation,
                                       crossvalidation_kwargs=dict(splits=5, kfold=True, split_coord='stimulus_id',
                                                                   stratification_coord='sentence_id'))
    return _Fedorenko2016(identifier='Fedorenko2016-encoding-v1', metric=metric)

# The new Fedorenko2016 benchmark (version 2)
def Fedorenko2016EncodingV2():
    benchmark = Fedorenko2016EncodingV1()
    benchmark._target_assembly = load_Fedorenko2016(electrodes='language', version=2) 
    benchmark.identifier = 'Fedorenko2016-encoding-v2'
    return benchmark

def Fedorenko2016AllEncodingV1():
    benchmark = Fedorenko2016EncodingV1()
    benchmark._target_assembly = load_Fedorenko2016(electrodes='all', version=1)
    benchmark.identifier = 'Fedorenko2016all-encoding-v1'
    return benchmark

def Fedorenko2016AllEncodingV2():
    benchmark = Fedorenko2016EncodingV1()
    benchmark._target_assembly = load_Fedorenko2016(electrodes='all', version=2)
    benchmark.identifier = 'Fedorenko2016all-encoding-v2'
    return benchmark

def Fedorenko2016NonLangEncoding():
    benchmark = Fedorenko2016EncodingV1()
    benchmark._target_assembly = load_Fedorenko2016(electrodes='non-language', version=2) # Version 2 - do not z-score in ecog_greta.py
    benchmark.identifier = 'Fedorenko2016nonlang-encoding'
    return benchmark


def Fedorenko2016RDM():
    metric = RDMCrossValidated(
        comparison_coord='stimulus_id',
        crossvalidation_kwargs=dict(split_coord='stimulus_id', stratification_coord='sentence_id',
                                    # doesn't work because train_size is deemed too small.
                                    # even though `train` is not needed, CrossValidation still splits it that way
                                    splits=5, kfold=True, test_size=None))
    return _Fedorenko2016(identifier='Fedorenko2016-rdm', metric=metric)


def holdout_subject_ceiling(assembly, metric, subject_column='subject'):
    subjects = set(assembly[subject_column].values)
    scores = []
    for subject in tqdm(subjects, desc='heldout subject'):
        subject_assembly = assembly[{'neuroid': [subject_value == subject
                                                 for subject_value in assembly[subject_column].values]}]
        # run subject pool as neural candidate
        subject_pool = subjects - {subject}
        pool_assembly = assembly[{'neuroid': [subject in subject_pool for subject in assembly[subject_column].values]}]
        score = metric(pool_assembly, subject_assembly)
        # store scores
        score = score.expand_dims(subject_column, _apply_raw=False)
        score.__setitem__(subject_column, [subject], _apply_raw=False)
        scores.append(score)

    scores = Score.merge(*scores)
    error = scores.sel(aggregation='center').std(subject_column)
    scores = apply_aggregate(lambda scores: scores.mean(subject_column), scores)
    scores.loc[{'aggregation': 'error'}] = error
    return scores


@store(identifier_ignore=['assembly', 'metric', 'subject_column'])
def extrapolation_ceiling(identifier, assembly, metric, subject_column='subject'):
    subjects = set(assembly[subject_column].values)
    scores = []
    for num_subjects in trange(2, len(subjects) + 1, desc='num subjects'):
        for sub_subjects in itertools.combinations(subjects, num_subjects):
            sub_assembly = assembly[{'neuroid': [subject in sub_subjects for subject in
                                                 assembly[subject_column].values]}]
            score = holdout_subject_ceiling(assembly=sub_assembly, metric=metric, subject_column=subject_column)
            score = score.expand_dims('num_subjects').expand_dims('sub_subjects')
            score['num_subjects'] = [num_subjects]
            score['sub_subjects'] = [str(sub_subjects)]
            scores.append(score)
    scores = Score.merge(*scores)
    return scores


def ceiling_normalize(score, benchmark_identifier):
    score = aggregate(score)
    benchmark = benchmark_pool[benchmark_identifier]()
    ceiling = benchmark.ceiling
    normalized_score = score.copy()
    normalized_center, normalized_error = ceiling_normalize_score_error(
        score.sel(aggregation='center').values, score.sel(aggregation='error').values,
        ceiling.sel(aggregation='center').values)
    normalized_score.loc[{'aggregation': 'center'}] = normalized_center
    normalized_score.loc[{'aggregation': 'error'}] = normalized_error
    return normalized_score


def ceiling_normalize_score_error(score, error, ceiling):
    return score / ceiling, error / ceiling


def aggregate(score):
    if hasattr(score, 'experiment'):
        score = score.mean('experiment')
    if hasattr(score, 'atlas'):
        score = score.mean('atlas')
    if hasattr(score, 'layer'):
        max_score = score.sel(aggregation='center').max()
        max_score = score[{'layer': (score.sel(aggregation='center') == max_score).values}].squeeze('layer', drop=True)
        max_score.attrs['raw'] = score.copy()
        score = max_score
    return score


benchmark_pool = {
    'stories_voxel_bold4s-encoding': StoriesVoxelEncoding,
    'stories_froi_bold4s-encoding': StoriesfROIEncoding,
    'stories_froi_bold4s-rdm': StoriesfROIRDM,
    'rdm': StoriesRDMBenchmark,
    'Pereira2018-encoding': PereiraEncoding,
    'Pereira2018ICA-encoding': PereiraICAEncoding,
    'Pereira2018Demean-encoding': PereiraDemeanEncoding,
    'Pereira2018Novisaud-encoding': PereiraNovisaudEncoding,
    'Pereira2018_languageresiduals-encoding': PereiraLanguageResidualsEncoding,
    'Pereira2018-encoding-min': PereiraEncodingMin,
    'Pereira2018-decoding': PereiraDecoding,
    'Pereira2018-rdm': PereiraRDM,
    'Fedorenko2016-rdm': Fedorenko2016RDM,
    'Fedorenko2016-encoding-v1': Fedorenko2016EncodingV1,
    'Fedorenko2016-encoding-v2': Fedorenko2016EncodingV2,
    'Fedorenko2016all-encoding-v1': Fedorenko2016AllEncodingV1,
    'Fedorenko2016all-encoding-v2': Fedorenko2016AllEncodingV2,
    'Fedorenko2016nonlang-encoding': Fedorenko2016NonLangEncoding,
}