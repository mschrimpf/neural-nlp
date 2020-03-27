import itertools
import logging
import numpy as np
from brainio_collection.fetch import fullname
from numpy.random.mtrand import RandomState
from scipy.optimize import curve_fit
from tqdm import tqdm, trange

from brainscore.metrics import Score
from brainscore.metrics.transformations import apply_aggregate
from result_caching import store


class HoldoutSubjectCeiling:
    def __init__(self, subject_column):
        self.subject_column = subject_column
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        scores = []
        iterate_subjects = self.get_subject_iterations(subjects)
        for subject in tqdm(iterate_subjects, desc='heldout subject'):
            try:
                subject_assembly = assembly[{'neuroid': [subject_value == subject
                                                         for subject_value in assembly[self.subject_column].values]}]
                # run subject pool as neural candidate
                subject_pool = subjects - {subject}
                pool_assembly = assembly[
                    {'neuroid': [subject in subject_pool for subject in assembly[self.subject_column].values]}]
                score = self.score(pool_assembly, subject_assembly, metric=metric)
                # store scores
                score = score.expand_dims(self.subject_column, _apply_raw=False)
                score.__setitem__(self.subject_column, [subject], _apply_raw=False)
                scores.append(score)
            except NoOverlapException as e:
                self._logger.debug(f"Ignoring no overlap {e}")
                continue  # ignore
            except ValueError as e:
                if "Found array with" in str(e):
                    self._logger.debug(f"Ignoring empty array {e}")
                    continue
                else:
                    raise e

        scores = Score.merge(*scores)
        error = scores.sel(aggregation='center').std(self.subject_column)
        scores = apply_aggregate(lambda scores: scores.mean(self.subject_column), scores)
        scores.loc[{'aggregation': 'error'}] = error
        return scores

    def get_subject_iterations(self, subjects):
        return subjects  # iterate over all subjects

    def score(self, pool_assembly, subject_assembly, metric):
        return metric(pool_assembly, subject_assembly)


class ExtrapolationCeiling:
    def __init__(self, subject_column='subject'):
        self.subject_column = subject_column
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self._logger = logging.getLogger(fullname(self))

    @store(identifier_ignore=['assembly', 'metric'])
    def __call__(self, identifier, assembly, metric):
        scores = self.collect(identifier, assembly=assembly, metric=metric)
        return self.extrapolate(scores)

    def collect(self, identifier, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        subject_subsamples = self.build_subject_subsamples(subjects)
        scores = []
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
            for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):
                try:
                    score = self.holdout_ceiling(assembly=sub_assembly, metric=metric)
                    score = score.expand_dims('num_subjects')
                    score['num_subjects'] = [num_subjects]
                    for key, selection in selections.items():
                        expand_dim = f'sub_{key}'
                        score = score.expand_dims(expand_dim)
                        score[expand_dim] = [str(selection)]
                    scores.append(score.raw)
                except KeyError as e:  # nothing to merge
                    if str(e) == "'z'":
                        self._logger.debug(f"Ignoring merge error {e}")
                        continue
                    else:
                        raise e
        scores = Score.merge(*scores)
        ceilings = self.average_collected(scores)
        ceilings.attrs['raw'] = scores
        return ceilings

    def build_subject_subsamples(self, subjects):
        return tuple(range(2, len(subjects) + 1))

    def iterate_subsets(self, assembly, num_subjects):
        subjects = set(assembly[self.subject_column].values)
        subject_combinations = list(itertools.combinations(subjects, num_subjects))
        for sub_subjects in subject_combinations:
            sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                 for subject in assembly[self.subject_column].values]}]
            yield {self.subject_column: sub_subjects}, sub_assembly

    def average_collected(self, scores):
        return scores.median('neuroid')

    def extrapolate(self, ceilings):
        def v(x, v0, tau0):
            return v0 * (1 - np.exp(-x / tau0))

        # figure out how many extrapolation x points we have. E.g. for Pereira, also not all combinations are possible
        subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
        num_bootstraps = 100
        rng = RandomState(0)
        bootstrap_params = []
        for _ in trange(num_bootstraps, desc='bootstraps'):
            bootstrapped_scores = []
            for num_subjects in subject_subsamples:
                num_scores = ceilings.sel(num_subjects=num_subjects)
                # the sub_subjects dimension creates nans, get rid of those
                num_scores = num_scores.dropna(f'sub_{self.subject_column}')
                assert set(num_scores.dims) == {f'sub_{self.subject_column}', 'split'}
                # choose from subject subsets and the splits therein, with replacement for variance
                choices = num_scores.values.flatten()
                bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                bootstrapped_scores.append(np.mean(bootstrapped_score))

            try:
                params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                         # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                         bounds=([0, -np.inf], [1, np.inf]))
                bootstrap_params.append(params)
            except RuntimeError:  # optimal parameters not found
                continue
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params])
        median_ys = np.median(ys, axis=0)
        diffs = np.diff(median_ys)
        end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
        # put together
        center = np.median(np.array(bootstrap_params)[:, 0])
        error = ci_error(ys[:, end_x], center=center)
        score = Score([center] + list(error),
                      coords={'aggregation': ['center', 'error_low', 'error_high']}, dims=['aggregation'])
        score.attrs['raw'] = ceilings
        score.attrs['bootstrapped_params'] = bootstrap_params
        score.attrs['endpoint_x'] = end_x
        return score


def ci_error(samples, center, confidence=.95):
    low, high = 100 * (1 - confidence) / 2, 100 * 1 - ((1 - confidence) / 2)
    confidence_below, confidence_above = np.percentile(samples, low), np.percentile(samples, high)
    confidence_below, confidence_above = center - confidence_below, confidence_above - center
    return confidence_below, confidence_above


class NoOverlapException(Exception):
    pass
