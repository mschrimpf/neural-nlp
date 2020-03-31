import fire
import itertools
import logging
import numpy as np
import seaborn
import sys
from brainio_base.assemblies import walk_coords
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from numpy.random.mtrand import RandomState
from pathlib import Path
from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.regression import pearsonr_correlation
from brainscore.metrics.transformations import apply_aggregate
from neural_nlp import benchmark_pool
from neural_nlp.benchmarks.ceiling import ci_error, v
from neural_nlp.neural_data.fmri import load_voxels
from result_caching import store

_logger = logging.getLogger(__name__)


@store()
def fROI_correlation():
    assembly = load_voxels()

    stories = list(sorted(set(assembly['story'].values)))
    subjects = list(sorted(set(assembly['subject_UID'].values)))
    split_scores = []
    correlate = pearsonr_correlation(xarray_kwargs=dict(correlation_coord='stimulus_id', neuroid_coord='fROI_area'))
    cross_stories_subjects = list(itertools.product(stories, subjects))
    for story, heldout_subject in tqdm(cross_stories_subjects, desc='cross-{story,subject}'):
        story_assembly = assembly[{'presentation': [coord_story == story for coord_story in assembly['story'].values]}]
        subject_pool = story_assembly[{'neuroid': [subject != heldout_subject
                                                   for subject in story_assembly['subject_UID'].values]}]
        subject_pool = average_subregions(subject_pool)
        heldout = story_assembly[{'neuroid': [subject == heldout_subject
                                              for subject in story_assembly['subject_UID'].values]}]
        heldout = average_subregions(heldout)
        split_score = correlate(subject_pool, heldout)
        split_score = type(split_score)(split_score.values, coords={
            coord: (dims, values) for coord, dims, values in walk_coords(split_score)
            if not coord.startswith('subject_') and coord != 'neuroid_id'}, dims=split_score.dims)

        split_score = split_score.expand_dims('heldout_subject').expand_dims('story')
        split_score['heldout_subject'], split_score['story'] = [heldout_subject], [story]
        split_scores.append(split_score)
    correlation = Score.merge(*split_scores)

    correlation = apply_aggregate(lambda scores: scores.mean('neuroid').mean('story'), correlation)
    center = correlation.mean('heldout_subject')
    error = correlation.std('heldout_subject')
    score = Score([center, error], coords={**{'aggregation': ['center', 'error']},
                                           **{coord: (dims, values) for coord, dims, values in walk_coords(center)}},
                  dims=('aggregation',) + center.dims)
    score.attrs[Score.RAW_VALUES_KEY] = correlation.attrs[Score.RAW_VALUES_KEY]
    return score


def average_subregions(assembly):
    del assembly['threshold']
    assembly = assembly.multi_dim_apply(['stimulus_id', 'fROI_area'], lambda group, **_: group.mean())
    _, index = np.unique(assembly['fROI_area'], return_index=True)
    assembly = assembly.isel(neuroid=index)
    return assembly


def plot_extrapolation_ceiling(benchmark='Fedorenko2016-encoding'):
    benchmark_impl = benchmark_pool[benchmark]
    ceiling = benchmark_impl.ceiling

    fig, ax = pyplot.subplots()

    # plot actual data splits
    raw_ceilings = ceiling.raw.raw
    subject_columns = prefixdict(default='sub_subject_UID', Pereira='sub_subject', stories='sub_subject_id')
    subject_column = subject_columns[benchmark]
    num_splits = raw_ceilings.stack(numsplit=['num_subjects', subject_column, 'split'])
    jitter = .25
    rng = RandomState(0)
    x = num_splits['num_subjects'].values + (-jitter / 2 + jitter * rng.rand(len(num_splits['num_subjects'])))
    y = num_splits.median('neuroid').values
    ax.scatter(x, y, color='black', s=1, zorder=10)

    # bootstrap and average fits
    x = np.arange(0, max(ceiling.endpoint_x, max(raw_ceilings['num_subjects'].values)) + 2)
    ys = np.array([v(x, *params) for params in ceiling.bootstrapped_params.values])
    for y in ys:
        ax.plot(x, y, alpha=.05, color='gray')
    median_ys = np.median(ys, axis=0)
    error = confidence_interval(ys.T, centers=median_ys)
    ax.errorbar(x=x, y=median_ys, yerr=list(zip(*error)), linestyle='dashed', color='gray')
    estimated_ceiling = ceiling.sel(aggregation='center').values
    ax.text(.65, .1, s=f"asymptote {estimated_ceiling :.2f} at #~{ceiling.endpoint_x.values.tolist()}",
            ha='center', va='center', transform=ax.transAxes)

    # plot meta
    ax.set_title(benchmark)
    ax.set_xlabel('# subjects')
    ax.set_ylabel('estimated ceiling')
    ax.set_ylim([0.9 * np.nanmin(num_splits.values),
                 (1.8 if benchmark.startswith('Fedorenko') else 1.2) * np.nanmax(num_splits.values)])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    savepath = Path(__file__).parent / f'extrapolation-{benchmark}.png'
    _logger.debug(f"Saving to {savepath}")
    fig.savefig(savepath)


def confidence_interval(data, centers, confidence=0.95):
    assert len(data) == len(centers)
    cis = []
    for samples, center in zip(data, centers):
        confidence_below, confidence_above = ci_error(samples, center, confidence)
        cis.append((confidence_below, confidence_above))
    return cis


def plot_ceiling_subsamples(benchmark='Fedorenko2016-encoding'):
    benchmark_impl = benchmark_pool[benchmark]
    ceilings = benchmark_impl.ceiling
    raw_ceilings = ceilings.raw.median('neuroid')
    num_splits = raw_ceilings.stack(numsplit=['num_subjects', 'sub_subjects'])
    fig, axes = pyplot.subplots(ncols=len(np.unique(num_splits['num_subjects'])))
    for ax, ns in zip(axes.flatten(), np.unique(num_splits['num_subjects'])):
        ax.hist(num_splits.sel(num_subjects=ns).values.flatten())
        ax.set_xlabel(f"{ns}")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f'hist-{benchmark}.png')


class prefixdict(dict):
    def __init__(self, default=None, **kwargs):
        super(prefixdict, self).__init__(**kwargs)
        self._default = default

    def __getitem__(self, item):
        subitem = item
        while len(subitem) > 1:
            try:
                return super(prefixdict, self).__getitem__(subitem)
            except KeyError:
                subitem = subitem[:-1]
        return self._default


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore')
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
