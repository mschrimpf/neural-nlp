import fire
import numpy as np
from matplotlib import pyplot
from pathlib import Path

from neural_nlp.analyze import savefig
from neural_nlp import benchmark_pool


def plot_nans():
    benchmark = benchmark_pool['Futrell2018-encoding']
    assembly = benchmark._target_assembly
    assembly = assembly.transpose('neuroid', 'presentation')
    nans = np.isnan(assembly)

    fig, ax = pyplot.subplots(figsize=(50, 5))
    ax.imshow(nans)
    xtick_frequency, ytick_frequency = 25, 20
    xticks = np.arange(0, len(nans['word']), xtick_frequency)
    ax.set_xticks(xticks)
    ax.set_xticklabels(nans['word'].values[xticks], rotation=90)
    yticks = np.arange(0, len(nans['subject_id']), ytick_frequency)
    ax.set_yticks(yticks)
    ax.set_yticklabels(nans['subject_id'].values[yticks])
    savefig(Path(__file__).parent / "reading_times-nans.png")


def plot_histogram(datapoint_cutoff=None):
    benchmark = benchmark_pool['Futrell2018-encoding']
    assembly = benchmark._target_assembly
    non_nans = ~np.isnan(assembly)
    sums = non_nans.sum('presentation')
    if datapoint_cutoff is not None:
        sums = sums[sums < datapoint_cutoff]

    fig, ax = pyplot.subplots()
    ax.hist(sums, bins=100)
    ax.set_xlabel('number of data points (not nan)')
    ax.set_ylabel('number of subjects')
    savefig(fig, Path(__file__).parent / ("reading_times-nans_hist" +
                                          (f'-{datapoint_cutoff}' if datapoint_cutoff is not None else '') + ".png"))


if __name__ == '__main__':
    fire.Fire()
