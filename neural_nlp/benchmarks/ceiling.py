import itertools
import logging
import numpy as np
from brainio_base.assemblies import DataAssembly, merge_data_arrays, walk_coords, array_is_element
from brainio_collection.fetch import fullname
from numpy import AxisError
from numpy.random.mtrand import RandomState
from scipy.optimize import curve_fit
from tqdm import tqdm, trange

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
                apply_raw = 'raw' in score.attrs and \
                            not hasattr(score.raw, self.subject_column)  # only propagate if column not part of score
                score = score.expand_dims(self.subject_column, _apply_raw=apply_raw)
                score.__setitem__(self.subject_column, [subject], _apply_raw=apply_raw)
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
    def __init__(self, subject_column='subject', extrapolation_dimension='neuroid',
                 num_bootstraps=100, post_process=None):
        self._logger = logging.getLogger(fullname(self))
        self.subject_column = subject_column
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self.extrapolation_dimension = extrapolation_dimension
        self.num_bootstraps = num_bootstraps
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
        for i in trange(len(ceilings[self.extrapolation_dimension]),
                        desc=f'{self.extrapolation_dimension} extrapolations'):
            try:
                # extrapolate per-neuroid ceiling
                neuroid_ceiling = ceilings.isel(**{self.extrapolation_dimension: [i]})
                extrapolated_ceiling = self.extrapolate_neuroid(neuroid_ceiling.squeeze())
                extrapolated_ceiling = self.add_neuroid_meta(extrapolated_ceiling, neuroid_ceiling)
                neuroid_ceilings.append(extrapolated_ceiling)
                # also keep track of bootstrapped parameters
                neuroid_bootstrap_params = extrapolated_ceiling.bootstrapped_params
                neuroid_bootstrap_params = self.add_neuroid_meta(neuroid_bootstrap_params, neuroid_ceiling)
                bootstrap_params.append(neuroid_bootstrap_params)
                # and endpoints
                endpoint_x = self.add_neuroid_meta(extrapolated_ceiling.endpoint_x, neuroid_ceiling)
                endpoint_xs.append(endpoint_x)
            except AxisError:  # no extrapolation successful (happens for 1 neuroid in Pereira)
                continue
        # merge and add meta
        self._logger.debug("Merging neuroid ceilings")
        neuroid_ceilings = manual_merge(*neuroid_ceilings, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['raw'] = ceilings
        self._logger.debug("Merging bootstrap params")
        bootstrap_params = manual_merge(*bootstrap_params, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['bootstrapped_params'] = bootstrap_params
        self._logger.debug("Merging endpoints")
        endpoint_xs = manual_merge(*endpoint_xs, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['endpoint_x'] = endpoint_xs
        # aggregate
        ceiling = self.aggregate_neuroid_ceilings(neuroid_ceilings)
        return ceiling

    def add_neuroid_meta(self, target, source):
        target = target.expand_dims(self.extrapolation_dimension)
        for coord, dims, values in walk_coords(source):
            if array_is_element(dims, self.extrapolation_dimension):
                target[coord] = dims, values
        return target

    def aggregate_neuroid_ceilings(self, neuroid_ceilings):
        ceiling = neuroid_ceilings.median(self.extrapolation_dimension)
        ceiling.attrs['bootstrapped_params'] = neuroid_ceilings.bootstrapped_params.median(self.extrapolation_dimension)
        ceiling.attrs['endpoint_x'] = neuroid_ceilings.endpoint_x.median(self.extrapolation_dimension)
        ceiling.attrs['raw'] = neuroid_ceilings
        return ceiling

    def extrapolate_neuroid(self, ceilings):
        # figure out how many extrapolation x points we have. E.g. for Pereira, not all combinations are possible
        subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
        rng = RandomState(0)
        bootstrap_params = []
        for bootstrap in range(self.num_bootstraps):
            bootstrapped_scores = []
            for num_subjects in subject_subsamples:
                num_scores = ceilings.sel(num_subjects=num_subjects)
                # the sub_subjects dimension creates nans, get rid of those
                num_scores = num_scores.dropna(f'sub_{self.subject_column}')
                assert set(num_scores.dims) == {f'sub_{self.subject_column}', 'split'} or \
                       set(num_scores.dims) == {f'sub_{self.subject_column}'}
                # choose from subject subsets and the splits therein, with replacement for variance
                choices = num_scores.values.flatten()
                bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                bootstrapped_scores.append(np.mean(bootstrapped_score))

            try:
                params = self.fit(subject_subsamples, bootstrapped_scores)
            except RuntimeError:  # optimal parameters not found
                params = [np.nan, np.nan]
            params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                  dims=['bootstrap', 'param'])
            bootstrap_params.append(params)
        bootstrap_params = merge_data_arrays(bootstrap_params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                       if not np.isnan(params).any()])
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
        score.attrs['endpoint_x'] = DataAssembly(end_x)
        return score

    def fit(self, subject_subsamples, bootstrapped_scores):
        params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                 # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                 bounds=([0, -np.inf], [1, np.inf]))
        return params

    def post_process(self, scores):
        if self._post_process is not None:
            scores = self._post_process(scores)
        return scores


def ci_error(samples, center, confidence=.95):
    low, high = 100 * ((1 - confidence) / 2), 100 * (1 - ((1 - confidence) / 2))
    confidence_below, confidence_above = np.nanpercentile(samples, low), np.nanpercentile(samples, high)
    confidence_below, confidence_above = center - confidence_below, confidence_above - center
    return confidence_below, confidence_above


def manual_merge(*elements, on='neuroid'):
    dims = elements[0].dims
    assert all(element.dims == dims for element in elements[1:])
    merge_index = dims.index(on)
    # the coordinates in the merge index should have the same keys
    assert _coords_match(elements, dim=on,
                         match_values=False), f"coords in {[element[on] for element in elements]} do not match"
    # all other dimensions, their coordinates and values should already align
    for dim in set(dims) - {on}:
        assert _coords_match(elements, dim=dim,
                             match_values=True), f"coords in {[element[dim] for element in elements]} do not match"
    # merge values without meta
    merged_values = np.concatenate([element.values for element in elements], axis=merge_index)
    # piece together with meta
    result = type(elements[0])(merged_values, coords={
        **{coord: (dims, values)
           for coord, dims, values in walk_coords(elements[0])
           if not array_is_element(dims, on)},
        **{coord: (dims, np.concatenate([element[coord].values for element in elements]))
           for coord, dims, _ in walk_coords(elements[0])
           if array_is_element(dims, on)}}, dims=elements[0].dims)
    return result


def _coords_match(elements, dim, match_values=False):
    first_coords = [(key, tuple(value)) if match_values else key for _, key, value in walk_coords(elements[0][dim])]
    other_coords = [[(key, tuple(value)) if match_values else key for _, key, value in walk_coords(element[dim])]
                    for element in elements[1:]]
    return all(tuple(first_coords) == tuple(coords) for coords in other_coords)


class NoOverlapException(Exception):
    pass
