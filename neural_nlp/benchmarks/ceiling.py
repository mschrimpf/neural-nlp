import itertools
import numpy as np
import scipy.stats
from numpy.random.mtrand import RandomState
from scipy.optimize import curve_fit
from tqdm import tqdm, trange

from brainscore.metrics import Score
from brainscore.metrics.transformations import apply_aggregate
from result_caching import store


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


class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


class ExtrapolationCeiling:
    def __init__(self, subject_column='subject'):
        self.subject_column = subject_column

    @store(identifier_ignore=['assembly', 'metric'])
    def __call__(self, identifier, assembly, metric):
        scores = self.collect(identifier, assembly=assembly, metric=metric)
        ceilings = self.average_collected(scores)
        ceilings.attrs['raw'] = scores
        return self.extrapolate(ceilings)

    def collect(self, identifier, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        subject_subsamples = list(range(2, len(subjects) + 1))
        scores = []
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            selection_combinations = self.iterate_subsets(assembly, num_subjects=num_subjects)
            for selections, sub_assembly in tqdm(selection_combinations, desc='selections'):
                score = holdout_subject_ceiling(assembly=sub_assembly, metric=metric,
                                                subject_column=self.subject_column)
                score = score.expand_dims('num_subjects')
                score['num_subjects'] = [num_subjects]
                for key, selection in selections.items():
                    expand_dim = f'sub_{key}'
                    score = score.expand_dims(expand_dim)
                    score[expand_dim] = [str(selection)]
                scores.append(score.raw)
        scores = Score.merge(*scores)
        ceilings = self.average_collected(scores)
        ceilings.attrs['raw'] = scores
        return ceilings

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
        def v(x, v0, tau0, a):
            return v0 * (1 - np.exp((-x + a) / tau0))

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

            params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores)
            bootstrap_params.append(params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params])
        median_ys = np.median(ys, axis=0)
        diffs = np.diff(median_ys)
        end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
        # put together
        score = Score([median_ys[end_x], scipy.stats.median_absolute_deviation(ys[:, end_x])],
                      coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        score.attrs['raw'] = ceilings
        score.attrs['bootstrapped_params'] = bootstrap_params
        score.attrs['endpoint_x'] = end_x
        return score
