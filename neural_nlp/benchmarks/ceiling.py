import itertools
import logging
import numpy as np
from brainio_base.assemblies import DataAssembly, merge_data_arrays
from brainio_collection.fetch import fullname
from numpy.random.mtrand import RandomState
from scipy.optimize import curve_fit
from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.transformations import apply_aggregate
from result_caching import store


def v(x, v0, tau0):
    return v0 * (1 - np.exp(-x / tau0))


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
    def __init__(self, subject_column='subject', post_process=None):
        self.subject_column = subject_column
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self._logger = logging.getLogger(fullname(self))
        self._post_process = post_process

    @store(identifier_ignore=['assembly', 'metric'])
    def __call__(self, identifier, assembly, metric):
        scores = self.collect(identifier, assembly=assembly, metric=metric)
        return self.extrapolate(scores)

    @store(identifier_ignore=['assembly', 'metric'])
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
        scores = self.post_process(scores)
        return scores

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
        neuroid_ceilings, bootstrap_params, endpoint_xs = [], [], []
        for i, neuroid_id in enumerate(tqdm(ceilings['neuroid_id'].values, desc='neuroid extrapolations')):
            # extrapolate per-neuroid ceiling
            neuroid_ceiling = ceilings.isel(neuroid=i)
            extrapolated_ceiling = self.extrapolate_neuroid(neuroid_ceiling)
            extrapolated_ceiling = extrapolated_ceiling.expand_dims('neuroid_id')
            extrapolated_ceiling['neuroid_id'] = [neuroid_id]
            neuroid_ceilings.append(extrapolated_ceiling)
            # also keep track of bootstrapped parameters
            neuroid_bootstrap_params = extrapolated_ceiling.bootstrapped_params
            neuroid_bootstrap_params = neuroid_bootstrap_params.expand_dims('neuroid_id')
            neuroid_bootstrap_params['neuroid_id'] = [neuroid_id]
            bootstrap_params.append(neuroid_bootstrap_params)
            # and endpoints
            endpoint_x = extrapolated_ceiling.endpoint_x
            endpoint_x = DataAssembly([endpoint_x], coords={'neuroid_id': [neuroid_id]}, dims=['neuroid_id'])
            endpoint_xs.append(endpoint_x)
        # merge and add meta
        neuroid_ceilings = Score.merge(*neuroid_ceilings)
        neuroid_ceilings = neuroid_ceilings.stack(neuroid=['neuroid_id'])
        neuroid_ceilings.attrs['raw'] = ceilings
        bootstrap_params = merge_data_arrays(bootstrap_params)
        bootstrap_params = bootstrap_params.stack(neuroid=['neuroid_id'])
        neuroid_ceilings.attrs['bootstrapped_params'] = bootstrap_params
        endpoint_xs = merge_data_arrays(endpoint_xs)
        endpoint_xs = endpoint_xs.stack(neuroid=['neuroid_id'])
        neuroid_ceilings.attrs['endpoint_x'] = endpoint_xs
        # aggregate
        ceiling = neuroid_ceilings.median('neuroid')
        ceiling.attrs['bootstrapped_params'] = neuroid_ceilings.bootstrapped_params.median('neuroid')
        ceiling.attrs['endpoint_x'] = neuroid_ceilings.endpoint_x.median('neuroid')
        ceiling.attrs['raw'] = neuroid_ceilings
        return ceiling

    def extrapolate_neuroid(self, ceilings):
        # figure out how many extrapolation x points we have. E.g. for Pereira, not all combinations are possible
        subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
        num_bootstraps = 100
        rng = RandomState(0)
        bootstrap_params = []
        for bootstrap in range(num_bootstraps):
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
                params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                      dims=['bootstrap', 'param'])
                bootstrap_params.append(params)
            except RuntimeError:  # optimal parameters not found
                continue
        bootstrap_params = merge_data_arrays(bootstrap_params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values])
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

    def post_process(self, scores):
        if self._post_process is not None:
            scores = self._post_process(scores)
        return scores


def ci_error(samples, center, confidence=.95):
    low, high = 100 * (1 - confidence) / 2, 100 * 1 - ((1 - confidence) / 2)
    confidence_below, confidence_above = np.percentile(samples, low), np.percentile(samples, high)
    confidence_below, confidence_above = center - confidence_below, confidence_above - center
    return confidence_below, confidence_above


class NoOverlapException(Exception):
    pass
