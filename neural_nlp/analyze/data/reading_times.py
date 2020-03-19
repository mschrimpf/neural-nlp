import numpy as np
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import benchmark_pool


def plot_nans():
    benchmark = benchmark_pool['stories_readingtime-encoding']
    assembly = benchmark._target_assembly
    nans = np.isnan(assembly)

    fig, ax = pyplot.subplots(figsize=(5, 50))
    ax.imshow(nans)
    xtick_frequency, ytick_frequency = 20, 25
    xticks = np.arange(0, len(nans['subject_id']), xtick_frequency)
    ax.set_xticks(xticks)
    ax.set_xticklabels(nans['subject_id'].values[xticks], rotation=90)
    yticks = np.arange(0, len(nans['word']), ytick_frequency)
    ax.set_yticks(yticks)
    ax.set_yticklabels(nans['word'].values[yticks])
    fig.savefig(Path(__file__).parent / "reading_times-nans.png")


if __name__ == '__main__':
    plot_nans()
