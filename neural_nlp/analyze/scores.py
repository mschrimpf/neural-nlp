import fire
import seaborn
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import score

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


def compare(benchmark1='Pereira2018-encoding', benchmark2='Pereira2018-decoding'):
    scores1 = [score(benchmark=benchmark1, model=model) for model in models]
    scores2 = [score(benchmark=benchmark2, model=model) for model in models]
    x, xerr = [s.sel(aggregation='center') for s in scores1], [s.sel(aggregation='error') for s in scores1]
    y, yerr = [s.sel(aggregation='center') for s in scores2], [s.sel(aggregation='error') for s in scores2]
    fig, ax = pyplot.subplots()
    ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, fmt='.')
    for model, _x, _y in zip(models, x, y):
        ax.text(_x, _y, model)
    ax.set_xlabel(f"model scores on {benchmark1}")
    ax.set_ylabel(f"model scores on {benchmark2}")
    pyplot.savefig(Path(__file__).parent / 'scores' / f"{benchmark1}__{benchmark2}.png")
    return fig


if __name__ == '__main__':
    fire.Fire()
