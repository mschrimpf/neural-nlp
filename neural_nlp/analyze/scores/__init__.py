from decimal import Decimal

import fire
import itertools
import logging
import numpy as np
import pandas as pd
import scipy.stats
import seaborn
import sys
from matplotlib import pyplot
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm

from neural_nlp import score
from neural_nlp.analyze.sampled_architectures.neural_scores import score_all_models, \
    _score_model as score_architecture_model

logger = logging.getLogger(__name__)

models = ('glove',
          'lm_1b',
          'bert',
          'gpt2',
          'openaigpt',
          'transfoxl',
          'xlnet',
          'xlm', 'xlm-clm',
          'roberta')


def bar_models(benchmark='Pereira2018-encoding'):
    scores = [score(benchmark=benchmark, model=model) for model in models]
    fig, ax = _bars(models, scores, ylabel=f"model scores on {benchmark}")
    _savefig(fig, savename=benchmark)


def _bars(x, scores, ylabel=None):
    y, yerr = [s.sel(aggregation='center') for s in scores], [s.sel(aggregation='error') for s in scores]

    fig, ax = pyplot.subplots()
    ax.bar(x, y, yerr=yerr)
    ax.set_xticklabels(x, rotation=90, rotation_mode='anchor')
    ax.set_ylabel(ylabel)
    return fig, ax


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
    ax.text(ax.get_xlim()[1] * (0.8 if not flip_x else 1.2), ax.get_ylim()[0] * 1.1,
            (f"r={(correlation * (-1 if flip_x else 1)):.2f}" +
             '*' * max([i for i in range(5) if p <= 0.5 / (10 ** i)]))
            if p < 0.05 else f"n.s., p={p:.2f}")

    ax.set_xlabel(benchmark1)
    ax.set_ylabel(benchmark2)

    savepath = Path(__file__).parent / 'scores' / f"{benchmark1}__{benchmark2}.png"
    pyplot.savefig(savepath)
    logger.info(f"Saved to {savepath}")
    return fig


def sampled_architectures(zoo_dir='/braintree/data2/active/users/msch/zoo.wmt17-lm',
                          benchmark='Pereira2018-encoding-min'):
    scores = score_all_models(zoo_dir, benchmark=benchmark, perplexity_threshold=20)
    model_dirs, scores = list(scores.keys()), list(scores.values())
    architectures = [model_dir.name[:5] for model_dir in model_dirs]
    zoo_name = Path(zoo_dir).name
    fig, ax = _bars(architectures, scores, ylabel=f"MT model scores on {benchmark}")
    ax.set_ylim([ax.get_ylim()[0], 0.3])
    _savefig(fig, savename=f"{zoo_name}-{benchmark}")


def lstm_mt_vs_lm(benchmark='Pereira2018-encoding-min'):
    mt_score = score_architecture_model('/braintree/data2/active/users/msch/zoo.wmt17/'  # this is actually -mt
                                        'eb082aa158bb3ae2f9c5c3d4c5ff7bae2f93f901',
                                        benchmark=benchmark)
    lm_score = score(model='lm_1b', benchmark=benchmark)
    fig, ax = _bars(['MT: WMT\'17', 'LM: 1B'], [mt_score, lm_score], fig_kwargs=dict(figsize=(5, 5)))
    ax.set_ylim([ax.get_ylim()[0], 0.3])
    ax.set_title('LSTM trained on Machine Translation/Language Modeling')
    _savefig(fig, 'lstm_mt_lm')


def collect_scores(benchmark, models):
    store_file = Path(__file__).parent / f'scores-{benchmark}.csv'
    if store_file.is_file():
        return pd.read_csv(store_file)
    data = []
    for model in tqdm(models, desc='model scores'):
        model_scores = score(benchmark=benchmark, model=model)
        for experiment, atlas, layer in itertools.product(
                model_scores['experiment'].values, model_scores['atlas'].values, model_scores['layer'].values):
            current_score = model_scores.sel(atlas=atlas, experiment=experiment, layer=layer)
            data.append({'experiment': experiment, 'atlas': atlas, 'benchmark': benchmark,
                         'model': model, 'layer': layer,
                         'score': current_score.sel(aggregation='center').values.tolist(),
                         'error': current_score.sel(aggregation='error').values.tolist()})
    data = pd.DataFrame(data)
    data.to_csv(store_file)
    return data


def fmri_experiment_correlations():
    scores = collect_scores(benchmark='Pereira2018-encoding', models=models)
    experiment2_scores = scores[scores['experiment'] == '384sentences']
    experiment3_scores = scores[scores['experiment'] == '243sentences']
    r, p = pearsonr(experiment2_scores['score'], experiment3_scores['score'])
    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.errorbar(x=experiment2_scores['score'], xerr=experiment2_scores['error'],
                y=experiment3_scores['score'], yerr=experiment3_scores['error'],
                fmt=' ', alpha=.5)
    ax.plot(ax.get_xlim(), ax.get_ylim(), linestyle='dashed', color='black')
    ax.set_xlabel('scores on 384sentences')
    ax.set_ylabel('scores on 243sentences')
    ax.text(0.9, 0.1, "r: " + (f"{r:.2f}" if p < .05 else "n.s."), ha='center', va='center', transform=ax.transAxes)
    ticks = np.arange(0, 0.35, 0.05)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ticklabels = ["0" if tick == 0 else f"{tick:.1f}"[1:] if Decimal(f"{tick:.2f}") % Decimal(".1") == 0 else ""
                  for tick in ticks]
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    fig.tight_layout()
    _savefig(fig, 'fmri-correlations')


def _savefig(fig, savename):
    fig.tight_layout()
    savepath = Path(__file__).parent / f"{savename}.png"
    logger.info(f"Saving to {savepath}")
    fig.savefig(savepath)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
