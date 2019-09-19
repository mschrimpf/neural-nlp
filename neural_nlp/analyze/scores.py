import numpy as np
from numpy.polynomial.polynomial import polyfit
import logging
import sys

import fire
import scipy.stats
import seaborn
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import score

logger = logging.getLogger(__name__)
seaborn.set()

models = [
    'glove', 'lm_1b',
    'bert', 'openaigpt', 'gpt2', 'transfoxl', 'xlnet', 'xlm', 'roberta',
]


def bars(benchmark='Pereira2018-encoding'):
    scores = [score(benchmark=benchmark, model=model) for model in models]
    y, yerr = [s.sel(aggregation='center') for s in scores], [s.sel(aggregation='error') for s in scores]

    fig, ax = pyplot.subplots()
    ax.bar(models, y, yerr=yerr)
    ax.set_ylabel(f"model scores on {benchmark}")
    pyplot.savefig(Path(__file__).parent / 'scores' / f"{benchmark}.png")
    return fig


def compare(benchmark1='Pereira2018-encoding', benchmark2='Pereira2018-rdm', flip_x=False):
    scores1 = [score(benchmark=benchmark1, model=model) for model in models]
    scores2 = [score(benchmark=benchmark2, model=model) for model in models]

    def get_center_err(s):
        if hasattr(s, 'aggregation'):
            return s.sel(aggregation='center').values.tolist(), s.sel(aggregation='error').values.tolist()
        if hasattr(s, 'measure'):
            ppl = s.sel(measure='test_perplexity').values.tolist()
            return np.log(ppl), 0
        raise ValueError(f"Unknown score structure: {s}")

    x, xerr = [get_center_err(s)[0] for s in scores1], [get_center_err(s)[1] for s in scores1]
    y, yerr = [get_center_err(s)[0] for s in scores2], [get_center_err(s)[1] for s in scores2]
    fig, ax = pyplot.subplots()
    ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, fmt='.')
    for model, _x, _y in zip(models, x, y):
        ax.text(_x, _y, model)

    if flip_x:
        ax.set_xlim(list(reversed(ax.get_xlim())))

    correlation, p = scipy.stats.pearsonr(x, y)
    b, m = polyfit(x, y, 1)
    ax.plot(ax.get_xlim(), b + m * np.array(ax.get_xlim()))
    ax.text(ax.get_xlim()[1] * (0.9 if not flip_x else 1.1), ax.get_ylim()[0] * 1.1,
            f"r={correlation:.2f}" if p < 0.05 else 'r n.s.')

    ax.set_xlabel(f"model scores on {benchmark1}")
    ax.set_ylabel(f"model scores on {benchmark2}")

    savepath = Path(__file__).parent / 'scores' / f"{benchmark1}__{benchmark2}.png"
    pyplot.savefig(savepath)
    logger.info(f"Saved to {savepath}")
    return fig


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()
