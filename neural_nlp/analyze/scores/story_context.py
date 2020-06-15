import logging
import pandas as pd
import sys
from functools import reduce
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import benchmark_pool
from neural_nlp.analyze import savefig
from neural_nlp.analyze.scores import choose_best_scores, collect_scores


def plot_num_sentences(model='gpt2-xl'):
    fig, axes = pyplot.subplots(ncols=2)
    for ax, normalize in zip(axes, [True, False]):
        scores = []
        for num_sentences in [1, 3, 6, 9]:
            benchmark = f'Blank2014sentences{num_sentences}fROI-encoding' if num_sentences != 9 \
                else 'Blank2014fROI-encoding'  # standard benchmark is 9 sentences
            _scores = collect_scores(benchmark=benchmark, models=[model], normalize=normalize)
            _scores = choose_best_scores(_scores)
            _scores['num_sentences'] = num_sentences
            ceiling = benchmark_pool[benchmark].ceiling
            _scores['ceiling'] = ceiling.sel(aggregation='center')
            scores.append(_scores)
        scores = reduce(lambda left, right: pd.concat([left, right]), scores)
        ax.errorbar(x=scores['num_sentences'], y=scores['score'], yerr=scores['error'], label='scores')
        ax.set_xlim(list(reversed(ax.get_xlim())))  # flip x
        ax.set_xlabel("# sentences")
        ax.set_ylabel(("Normalized " if normalize else "") + f"Consistency ({model})")
        ax.set_title(("Ceiled" if normalize else "Raw") + " Scores")

    savefig(fig, savename=Path(__file__).parent / "story_context")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot_num_sentences()
