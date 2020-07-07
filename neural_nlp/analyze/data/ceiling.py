import fire
import itertools
import logging
import matplotlib
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
from neural_nlp.analyze import savefig
from neural_nlp.analyze.data import subject_columns
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


def plot_extrapolation_ceiling(benchmark='Pereira2018-encoding', ytick_formatting_frequency=None):
    benchmark_impl = benchmark_pool[benchmark]
    ceiling = benchmark_impl.ceiling

    fig, ax = pyplot.subplots()

    # plot actual data splits
    bootstrapped_params = ceiling.bootstrapped_params
    if benchmark.startswith('Futrell'):
        raw_ceilings = ceiling.raw
    elif benchmark.startswith('Pereira'):
        raw_ceilings = ceiling.raw.raw.sel(atlas='language')
        bootstrapped_params = ceiling.raw.bootstrapped_params.sel(atlas='language')
        bootstrapped_params = bootstrapped_params.median(benchmark_impl._ceiler.extrapolation_dimension)
    else:
        raw_ceilings = ceiling.raw.raw
    subject_column = "sub_" + subject_columns[benchmark]
    num_splits = raw_ceilings.stack(numsplit=['num_subjects', subject_column, 'split'])
    step_size = raw_ceilings['num_subjects'].values[1] - raw_ceilings['num_subjects'].values[0]
    jitter = .25 * step_size
    rng = RandomState(0)
    x = num_splits['num_subjects'].values + (-jitter / 2 + jitter * rng.rand(len(num_splits['num_subjects'])))
    y = num_splits.median(benchmark_impl._ceiler.extrapolation_dimension).values \
        if not benchmark.startswith('Futrell') else num_splits.values
    ax.scatter(x, y, color='black', s=1, zorder=10)

    # bootstrap and average fits
    x = np.arange(0, max(ceiling.endpoint_x, max(raw_ceilings['num_subjects'].values)) * 1.3)
    ys = np.array([v(x, *params) for params in bootstrapped_params.values])
    for y in ys:
        ax.plot(x, y, alpha=.05, color='gray')
    median_ys = np.nanmedian(ys, axis=0)
    error = confidence_interval(ys.T, centers=median_ys)
    ax.errorbar(x=x, y=median_ys, yerr=list(zip(*error)), linestyle='dashed', color='gray')
    estimated_ceiling = ceiling.sel(aggregation='center').values
    ax.text(.65, .1, s=f"asymptote {estimated_ceiling :.2f} at #~{ceiling.endpoint_x.values.tolist():.0f}",
            ha='center', va='center', transform=ax.transAxes)

    # plot meta
    ax.set_title(benchmark)
    ax.set_xlabel('# participants')
    ax.set_ylabel('estimated ceiling')
    ax.set_ylim([0.9 * np.min(y), (1.35 if not benchmark.startswith('Futrell') else 1.1) * np.max(y)])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if ytick_formatting_frequency:
        @matplotlib.ticker.FuncFormatter
        def score_formatter(score, pos):
            return f"{score:.{ytick_formatting_frequency}f}"[1:]  # strip "0" in front of e.g. "0.2"

        ax.yaxis.set_major_formatter(score_formatter)
    fig.tight_layout()
    savefig(fig, Path(__file__).parent / f'extrapolation-{benchmark}')


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
    savefig(Path(__file__).parent / f'hist-{benchmark}.png')


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore')
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger('brainscore.metrics').setLevel(logging.INFO)
    seaborn.set(context='talk')
    fire.Fire()
