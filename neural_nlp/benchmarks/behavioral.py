"""
Behavioral benchmarks to probe match of model outputs against human outputs.
"""

import logging
import numpy as np
import xarray
import xarray as xr
from brainio_base.assemblies import NeuroidAssembly
from brainio_collection.fetch import fullname
from numpy.random.mtrand import RandomState
from tqdm import tqdm

from brainscore.benchmarks import Benchmark
from brainscore.metrics import Score
from brainscore.metrics.regression import linear_regression, pearsonr_correlation, CrossRegressedCorrelation
from brainscore.metrics.transformations import CartesianProduct, apply_aggregate
from brainscore.utils import LazyLoad
from neural_nlp.benchmarks.ceiling import ExtrapolationCeiling, HoldoutSubjectCeiling, NoOverlapException
from neural_nlp.benchmarks.neural import read_words, consistency
from neural_nlp.neural_data.naturalStories import load_naturalStories


class Futrell2018Encoding(Benchmark):
    """
    predict individual human reading times of natural stories.

    data source:
        Futrell et al., International Conference on Language Resources and Evaluation (LREC) 2018
        http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf
    """

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
        # normalize by ceiling
        # Note that we normalize by an overall ceiling, so the scores per subject are not normalized wrt. that subject
        # and should thus not be used by themselves. Only the aggregate makes sense to report
        normalized_subject_scores = consistency(cross_scores.sel(aggregation='center'),
                                                self.ceiling.sel(aggregation='center'))
        score = normalized_subject_scores.median('subject_id')
        std = normalized_subject_scores.std('subject_id')
        std['aggregation'] = 'error'
        # the MultiIndex tends to mess things up, so we get rid of it here
        score, std = xr.DataArray(score).expand_dims('aggregation'), xr.DataArray(std).expand_dims('aggregation')
        score = Score(Score.merge(score, std))
        score.attrs['raw'] = cross_scores
        score.attrs['ceiling'] = self.ceiling
        return score

    @property
    def ceiling(self):
        return self._ceiler(identifier=self.identifier, assembly=self._target_assembly, metric=self._metric)

    class ManySubjectExtrapolationCeiling(ExtrapolationCeiling):
        def __init__(self, subject_column, *args, **kwargs):
            super(Futrell2018Encoding.ManySubjectExtrapolationCeiling, self).__init__(
                subject_column, *args, **kwargs)
            self._rng = RandomState(0)
            self._num_subsamples = 5
            self.holdout_ceiling = Futrell2018Encoding.SplitHalfPoolCeiling(subject_column=subject_column)

        def build_subject_subsamples(self, subjects):
            return tuple(range(2, len(subjects) + 1, 5))  # reduce computational cost by only using every 5th point

        def iterate_subsets(self, assembly, num_subjects):
            # there are 180 subjects which makes for millions of combinations.
            # to avoid this computational explosion, we choose only a subset of the possible subject sub-samples.
            subjects = set(assembly[self.subject_column].values)
            subject_combinations = self._random_combinations(subjects, num_subjects,
                                                             choice=self._num_subsamples, rng=self._rng)
            for sub_subjects in tqdm(subject_combinations, desc="subject combinations"):
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

        def extrapolate(self, ceilings):
            ceilings = ceilings.median('neuroid')  # same here, combine neuroids
            extrapolated_ceiling = self.extrapolate_neuroid(ceilings)
            return extrapolated_ceiling

        def fit(self, subject_subsamples, bootstrapped_scores):
            valid = ~np.isnan(bootstrapped_scores)
            return super(Futrell2018Encoding.ManySubjectExtrapolationCeiling, self).fit(
                np.array(subject_subsamples)[valid], np.array(bootstrapped_scores)[valid])

    class SplitHalfPoolCeiling(HoldoutSubjectCeiling):
        def __init__(self, *args, **kwargs):
            super(Futrell2018Encoding.SplitHalfPoolCeiling, self).__init__(*args, **kwargs)
            self._rng = RandomState(0)
            self._num_bootstraps = 3

        def __call__(self, assembly, metric):
            subjects = set(assembly[self.subject_column].values)
            scores = []
            for bootstrap in tqdm(range(self._num_bootstraps), desc='split-half bootstrap'):
                try:
                    half1 = self._rng.choice(list(subjects), size=len(subjects) // 2, replace=False)
                    half2 = subjects - set(half1)
                    half1_assembly = assembly[{'neuroid': [subject_value in half1
                                                           for subject_value in assembly[self.subject_column].values]}]
                    half2_assembly = assembly[{'neuroid': [subject_value in half2
                                                           for subject_value in assembly[self.subject_column].values]}]
                    # run half2 as neural candidate for half1
                    score = self.score(half2_assembly, half1_assembly, metric=metric)
                    # store scores
                    score = score.expand_dims("bootstrap", _apply_raw=False)
                    score.__setitem__("bootstrap", [bootstrap], _apply_raw=False)
                    scores.append(score)
                except NoOverlapException as e:
                    self._logger.debug(f"Ignoring no overlap ({e})")
                    continue  # ignore

            scores = Score.merge(*scores)
            error = scores.sel(aggregation='center').std("bootstrap")
            scores = apply_aggregate(lambda scores: scores.mean("bootstrap"), scores)
            scores.loc[{'aggregation': 'error'}] = error
            return scores

        def score(self, source_assembly, target_assembly, metric):
            # mean, drop nans
            source_assembly, source_subjects = self.mean_subjects(source_assembly)
            target_assembly, target_subjects = self.mean_subjects(target_assembly)
            source_assembly = source_assembly.dropna('presentation')
            target_assembly = target_assembly.dropna('presentation')
            # align to the same stimuli that are non-nan in both source and target
            source_assembly = source_assembly[{'presentation': [
                stimulus_id in target_assembly['stimulus_id'].values
                for stimulus_id in source_assembly['stimulus_id'].values]}]
            exception_suffix = f"for source subjects {source_subjects} against target subjects {target_subjects}"
            if len(source_assembly['presentation']) < 10:  # if not enough overlap: skip
                raise NoOverlapException(f"Only {len(source_assembly)} stimuli left {exception_suffix}")
            target_assembly = target_assembly[{'presentation': [
                stimulus_id in source_assembly['stimulus_id'].values
                for stimulus_id in target_assembly['stimulus_id'].values]}]
            if len(target_assembly['presentation']) < 10:  # if not enough overlap: skip
                raise NoOverlapException(f"Only {len(target_assembly)} stimuli left {exception_suffix}")
            return super(Futrell2018Encoding.SplitHalfPoolCeiling, self).score(
                pool_assembly=source_assembly, subject_assembly=target_assembly, metric=metric)

        def mean_subjects(self, assembly):
            subjects = assembly['subject_id'].values
            assembly = assembly.mean('neuroid').expand_dims('neuroid')
            assembly['neuroid_id'] = 'neuroid', [0]
            assembly['subject_id'] = 'neuroid', [','.join(subjects)]
            return assembly, subjects


class Futrell2018MeanEncoding(Benchmark):
    """
    predict mean human reading times of natural stories.

    data source:
        Futrell et al., International Conference on Language Resources and Evaluation (LREC) 2018
        http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf
    """

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


class Futrell2018StoriesEncoding(Futrell2018Encoding):
    def __init__(self, *args, **kwargs):
        super(Futrell2018StoriesEncoding, self).__init__(*args, **kwargs)
        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='word_id', neuroid_coord='subject_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='word_id', neuroid_coord='subject_id'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True, unique_split_values=True,
                                        split_coord='story_id', stratification_coord=None))

    def _load_assembly(self):
        assembly = super(Futrell2018StoriesEncoding, self)._load_assembly()

        # filter subjects that have done at least 5 stories. Otherwise, we cannot 5-fold cross-validate across stories
        def count_stories(subject_assembly):
            subject_assembly = subject_assembly.dropna('presentation')
            num_stories = len(np.unique(subject_assembly['story_id'].values))
            return xr.DataArray(num_stories)

        subject_stories = assembly.groupby('subject_id').apply(count_stories)
        keep_subjects = subject_stories[subject_stories >= 5]['subject_id'].values
        keep_subjects = set(keep_subjects) - {'A1I02VZ07MZB7F'}  # this subject only has one data point for story 8
        assembly = assembly[{'neuroid': [subject in keep_subjects for subject in assembly['subject_id'].values]}]
        return assembly


class Futrell2018SentencesEncoding(Futrell2018Encoding):
    def __init__(self, *args, **kwargs):
        super(Futrell2018SentencesEncoding, self).__init__(*args, **kwargs)
        regression = linear_regression(xarray_kwargs=dict(
            stimulus_coord='word_id', neuroid_coord='subject_id'))
        correlation = pearsonr_correlation(xarray_kwargs=dict(
            correlation_coord='word_id', neuroid_coord='subject_id'))
        self._metric = CrossRegressedCorrelation(
            regression=regression, correlation=correlation,
            crossvalidation_kwargs=dict(splits=5, kfold=True, unique_split_values=True,
                                        split_coord='sentence_id', stratification_coord=None))

    def _load_assembly(self):
        assembly = super(Futrell2018SentencesEncoding, self)._load_assembly()

        # filter subjects that have done at least 5 sentences. Otherwise, we cannot 5-fold cross-validate across
        def count_sentences(subject_assembly):
            subject_assembly = subject_assembly.dropna('presentation')
            num_sentences = len(np.unique(subject_assembly['sentence_id'].values))
            num_words_per_story = subject_assembly.groupby('sentence_id').apply(
                lambda sentence_assembly: xr.DataArray(len(sentence_assembly['presentation'])))
            return xr.DataArray(num_sentences >= 5 and all(num_words_per_story >= 2))

        keep_subjects = assembly.groupby('subject_id').apply(count_sentences)
        keep_subjects = keep_subjects[keep_subjects]['subject_id'].values
        assembly = assembly[{'neuroid': [subject in keep_subjects for subject in assembly['subject_id'].values]}]
        return assembly

benchmark_pool = [
    ('Futrell2018-encoding', Futrell2018Encoding),
    ('Futrell2018mean-encoding', Futrell2018MeanEncoding),
    ('Futrell2018stories-encoding', Futrell2018StoriesEncoding),
    ('Futrell2018sentences-encoding', Futrell2018SentencesEncoding),
]
benchmark_pool = {identifier: LazyLoad(lambda identifier=identifier, ctr=ctr: ctr(identifier=identifier))
                  for identifier, ctr in benchmark_pool}
