import logging
import xarray
from brainio_base.assemblies import NeuroidAssembly
from brainio_collection.fetch import fullname

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct
from brainscore.utils import LazyLoad
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
        # self._ceiler = ExtrapolationCeiling(subject_column='subject_UID')

    @property
    def identifier(self):
        return self._identifier

    def _load_assembly(self):
        assembly = load_naturalStories()
        # we're going to treat subjects as "neuroids" to make it easier for our metrics
        assembly = assembly.rename({'subjects': 'neuroid'})
        assembly['neuroid_id'] = 'neuroid', assembly['subject_id']
        assembly = NeuroidAssembly(assembly)
        return assembly

    @property
    def ceiling(self):
        return Score([1, 0], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])  # FIXME
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

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
        return cross_scores


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
        # self._ceiler = ExtrapolationCeiling(subject_column='subject_UID')

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
        return Score([1, 0], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])  # FIXME
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

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
