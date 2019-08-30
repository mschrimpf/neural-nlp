import fire
import seaborn
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import score

seaborn.set()


def bars(benchmark='Pereira2018-encoding-min', savename='scores/bars.png'):
    models = [
        'glove', 'lm_1b',
        'bert', 'openaigpt', 'gpt2', 'transfoxl', 'xlnet', 'xlm', 'roberta',
    ]
    scores = [score(benchmark=benchmark, model=model) for model in models]
    y, yerr = [s.sel(aggregation='center') for s in scores], [s.sel(aggregation='error') for s in scores]

    fig, ax = pyplot.subplots()
    ax.bar(models, y, yerr=yerr)
    pyplot.savefig(Path(__file__).parent / savename)
    return fig


if __name__ == '__main__':
    fire.Fire()
