import logging
import numpy as np
import xarray
from brainio_base.assemblies import NeuroidAssembly
from brainio_collection.fetch import fullname
from numpy.random.mtrand import RandomState

from brainscore.benchmarks import Benchmark
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling, NoOverlapException
from neural_nlp.benchmarks.neural import read_words
from neural_nlp.neural_data.naturalStories import load_naturalStories


class StoriesReadingTimeEncoding(Benchmark):
    def __init__(self, identifier):
        self._logger = logging.getLogger(fullname(self))
        self._identifier = identifier
        assembly = LazyLoad(self._load_assembly)
        self._target_assembly = assembly
        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='word_id', neuroid_coord='subject_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='word_id', neuroid_coord='subject_id'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True,
                                        split_coord='word_id', stratification_coord='sentence_id'))
        self._cross = CartesianProduct(dividers=['subject_id'])
        self._ceiler = self.ManySubjectExtrapolationCeiling(subject_column='subject_id')

    @property
    def identifier(self):
        return self._identifier

    def _load_assembly(self):
        assembly = load_naturalStories()
        # we're going to treat subjects as "neuroids" to make it easier for our metrics which mostly deal with neurons
        assembly = assembly.rename({'subjects': 'neuroid'})
        assembly['neuroid_id'] = 'neuroid', assembly['subject_id']
        assembly = NeuroidAssembly(assembly)
        # this one subject only has 6 reading times that are above threshold (i.e. not nan). That is not enough for
        # 5-fold cross-validation where we cannot compute correlation on a single data point.
        assembly = assembly[{'neuroid': [subject != 'A2VG5S4UL5UGRS' for subject in assembly['subject_id'].values]}]
        return assembly

    def _apply_cross(self, source_assembly, cross_assembly):
        # for several subjects there are many nans when their performance was below threshold. We here drop those words
        not_nan = ~xarray.ufuncs.isnan(cross_assembly)
        cross_assembly = cross_assembly[not_nan]
        neuroid, subject_id = cross_assembly['neuroid'].values.tolist(), cross_assembly['subject_id'].values.tolist()
        # drop neuroid coords since they interfere with subject_id setting following expand_dims
        cross_assembly = cross_assembly.drop('subject_id').drop('neuroid')
        cross_assembly = cross_assembly.expand_dims('neuroid')
        cross_assembly['neuroid_id'] = ('neuroid', [str(neuroid)])
        cross_assembly['subject_id'] = ('neuroid', [subject_id])
        # source_assembly might not have the exact same index structure, so we filter based on remaining stimulus_id
        source_assembly = source_assembly[{'presentation': [stimulus_id in cross_assembly['stimulus_id'].values
                                                            for stimulus_id in source_assembly['stimulus_id'].values]}]
        return self._metric(source_assembly, cross_assembly)

    def __call__(self, candidate):
        self._logger.info('Computing activations')
        model_activations = read_words(candidate, self._target_assembly.attrs['stimulus_set'], reset_column='story_id',
                                       copy_columns=('stimulus_id', 'word_id', 'sentence_id'))
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        self._logger.info('Scoring model')
        cross_scores = self._cross(self._target_assembly,
                                   apply=lambda cross_assembly: self._apply_cross(model_activations, cross_assembly))
        score = cross_scores.mean('subject_id')
        cross_subjects_std = cross_scores.sel(aggregation='center').std()
        score.__setitem__({'aggregation': score['aggregation'] == 'error'}, cross_subjects_std, _apply_raw=False)
        return score

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class ManySubjectExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(StoriesReadingTimeEncoding.ManySubjectExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._rng = RandomState(0)
            self._num_subsamples = 5
            self.holdout_ceiling = StoriesReadingTimeEncoding.AveragePoolHoldoutCeiling(subject_column=subject_column)

        def build_subject_subsamples(self, subjects):
            return tuple(range(2, len(subjects) + 1, 5))  # reduce computational cost by only using every 5th point

        def iterate_subsets(self, assembly, num_subjects):
            # there are 180 subjects which makes for millions of combinations.
            # to avoid this computational explosion, we choose only a subset of the possible subject sub-samples.
            subjects = set(assembly[self.subject_column].values)
            subject_combinations = self._random_combinations(subjects, num_subjects,
                                                             choice=self._num_subsamples, rng=self._rng)
            for sub_subjects in subject_combinations:
                sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                     for subject in assembly[self.subject_column].values]}]
                yield {self.subject_column: sub_subjects}, sub_assembly

        def _random_combinations(self, subjects, num_subjects, choice, rng):
            # following https://stackoverflow.com/a/55929159/2225200
            # building all `itertools.combinations` followed by `rng.choice` subsampling
            # would lead to >1 trillion initial samples.
            subjects = np.array(list(subjects))
            combinations = set()
            while len(combinations) < choice:
                elements = rng.choice(subjects, size=num_subjects, replace=False)
                combinations.add(tuple(elements))
            return combinations

    class AveragePoolHoldoutCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(StoriesReadingTimeEncoding.AveragePoolHoldoutCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)

        def get_subject_iterations(self, subjects):
            return [self._rng.choice(list(subjects))]  # use only a single subject

        def score(self, pool_assembly, subject_assembly, metric):
            # mean the pool (if we were to keep every subject, we'd have a lot of nans), drop nans
            pool_subjects = pool_assembly['subject_id'].values
            pool_assembly = pool_assembly.mean('neuroid')
            pool_assembly = pool_assembly.dropna('presentation')
            pool_assembly = pool_assembly.expand_dims('neuroid')
            pool_assembly['neuroid_id'] = 'neuroid', [0]
            pool_assembly['subject_id'] = 'neuroid', [",".join(pool_subjects)]
            subject_assembly = subject_assembly.dropna('presentation')
            # align to the same stimuli that are non-nan in both pool and subject
            pool_assembly = pool_assembly[{'presentation': [
                stimulus_id in subject_assembly['stimulus_id'].values
                for stimulus_id in pool_assembly['stimulus_id'].values]}]
            if len(pool_assembly['presentation']) < 10:  # if not enough overlap: skip
                raise NoOverlapException(f"Only {len(pool_assembly)} stimuli left "
                                         f"for pool {pool_subjects} "
                                         f"against subject {subject_assembly['subject_id'].values}")
            subject_assembly = subject_assembly[{'presentation': [
                stimulus_id in pool_assembly['stimulus_id'].values
                for stimulus_id in subject_assembly['stimulus_id'].values]}]
            if len(subject_assembly['presentation']) < 10:  # if not enough overlap: skip
                raise NoOverlapException(f"Only {len(subject_assembly)} stimuli left "
                                         f"for subject {subject_assembly['subject_id'].values} "
                                         f"against pool {pool_subjects}")
            return super(StoriesReadingTimeEncoding.AveragePoolHoldoutCeiling, self).score(
                pool_assembly=pool_assembly, subject_assembly=subject_assembly, metric=metric)


class StoriesReadingTimeMeanEncoding(Benchmark):
    def __init__(self, identifier):
        self._logger = logging.getLogger(fullname(self))
        self._identifier = identifier
        assembly = LazyLoad(self._load_assembly)
        self._target_assembly = assembly
        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='word_id', neuroid_coord='subject_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='word_id', neuroid_coord='subject_id'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True,
                                        split_coord='word_id', stratification_coord='sentence_id'))

    @property
    def identifier(self):
        return self._identifier

    def _load_assembly(self):
        assembly = load_naturalStories()
        stimulus_set = assembly.stimulus_set
        # we're going to treat subjects as "neuroids" to make it easier for our metrics
        assembly = assembly.mean('subjects')
        assembly = assembly.expand_dims('neuroid')
        assembly['neuroid_id'] = 'neuroid', [0]
        assembly['subject_id'] = 'neuroid', ['all']
        assembly = NeuroidAssembly(assembly)
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly

    @property
    def ceiling(self):
        raise NotImplementedError("since we're operating on subjects mean, "
                                  "we have no consistent way of computing the ceiling")

    def __call__(self, candidate):
        self._logger.info('Computing activations')
        model_activations = read_words(candidate, self._target_assembly.attrs['stimulus_set'], reset_column='story_id',
                                       copy_columns=('stimulus_id', 'word_id', 'sentence_id'))
        assert set(model_activations['stimulus_id'].values) == set(self._target_assembly['stimulus_id'].values)
        self._logger.info('Scoring model')
        return self._metric(model_activations, self._target_assembly)


benchmark_pool = [
    ('stories_readingtime-encoding', StoriesReadingTimeEncoding),
    ('stories_readingtime_mean-encoding', StoriesReadingTimeMeanEncoding),
]
benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
