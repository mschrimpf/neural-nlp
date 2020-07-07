import logging
import numpy as np
import pandas as pd
import sys
from functools import reduce
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import benchmark_pool
from neural_nlp.analyze import savefig
from neural_nlp.analyze.scores import choose_best_scores, collect_scores


def plot_num_sentences(model='gpt2-xl'):
    fig, ax = pyplot.subplots(figsize=(4.5, 5.5))
    for ax, normalize in zip([ax], [True]):
        scores = []
        sentence_nums = np.arange(1, 10, 2)
        for sentence_num in sentence_nums:
            benchmark = f'Blank2014sentence{sentence_num}fROI-encoding' if sentence_num != 9 \
                else 'Blank2014fROI-encoding'  # standard benchmark is 9 sentences
            _scores = collect_scores(benchmark=benchmark, models=[model], normalize=normalize)
            _scores = choose_best_scores(_scores)
            _scores['sentence_num'] = sentence_num
            ceiling = benchmark_pool[benchmark].ceiling
            _scores['ceiling'] = ceiling.sel(aggregation='center')
            scores.append(_scores)
        scores = reduce(lambda left, right: pd.concat([left, right]), scores)
        ax.errorbar(x=scores['sentence_num'], y=scores['score'], yerr=scores['error'], label='scores')
        ax.set_xticks(sentence_nums)
        ax.set_xlim(list(reversed(ax.get_xlim())))  # flip x
        ax.set_xlabel("sentence # in story")
        ax.set_ylabel(("Normalized " if normalize else "") + f"Predictivity ({model})")

    savefig(fig, savename=Path(__file__).parent / "story_context")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plot_num_sentences()
