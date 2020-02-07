import fire
import itertools
import numpy as np
import seaborn
from brainio_base.assemblies import walk_coords
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.optimize import curve_fit
from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.regression import pearsonr_correlation
from brainscore.metrics.transformations import apply_aggregate
from neural_nlp import benchmark_pool
from neural_nlp.benchmarks.neural import extrapolation_ceiling
from neural_nlp.neural_data.fmri import load_voxels
from result_caching import store


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
    benchmark_impl = benchmark_pool[benchmark]()
    ceilings = extrapolation_ceiling(identifier=benchmark, assembly=benchmark_impl._target_assembly,
                                     subject_column='subject' if benchmark.startswith('Pereira') else 'subject_UID',
                                     metric=benchmark_impl._metric)
    # work from raw values to treat sub_subjects and split all as splits and then median/std on those
    ceilings = ceilings.raw
    if hasattr(ceilings, 'neuroid'):  # not true for RDMs
        ceilings = ceilings.median('neuroid')
    ceilings = ceilings.stack(subsplit=['sub_subjects', 'split'])  # introduces lots of nans due to non-overlap subjects
    y, yerr = ceilings.mean('subsplit'), ceilings.std('subsplit')

    # extrapolation
    def v(x, v0, tau0):
        return v0 * (1 - np.exp(-x / tau0))

    params, pcov = curve_fit(v, ceilings['num_subjects'], y)
    asymptote_threshold = .0005
    for interpolation_x in range(999):
        if v(interpolation_x + 1, *params) - v(interpolation_x, *params) < asymptote_threshold:
            break
    extrapolation_x = np.arange(interpolation_x + 3)
    extrapolation_y = v(extrapolation_x, *params)
    # plot
    fig, ax = pyplot.subplots()
    ax.errorbar(ceilings['num_subjects'].values, y.values, yerr=yerr.values)
    ax.plot(extrapolation_x, extrapolation_y, linestyle='dashed', color='gray')
    ax.text(.5, .1, s=f"~asymptote {params[0]:.2f} at #={np.ceil(interpolation_x):.0f}",
            ha='center', va='center', transform=ax.transAxes)
    # plot meta
    ax.set_title(benchmark)
    ax.set_xlabel('# subjects')
    ax.set_ylabel('ceiling')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f'extrapolation-{benchmark}.png')


if __name__ == '__main__':
    seaborn.set(context='talk')
    fire.Fire()
