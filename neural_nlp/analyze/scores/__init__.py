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

model_colors = {
    'glove': 'violet',
    'lm_1b': 'goldenrod',
    'transfoxl': 'peru',
    'bert': 'orangered',
    'roberta': 'brown',
    'openaigpt': 'steelblue',
    'gpt2': 'c',
    'gpt2-medium': 'teal',
    'gpt2-large': 'darkslategray',
    # 'gpt2-xl': 'darkcyan',
    'xlnet': 'gray',
    'xlm': 'green',
    'xlm-clm': 'darkgreen',
}

models = tuple(model_colors.keys())

fmri_atlases = ('DMN', 'MD', 'language', 'auditory', 'visual')

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
        if hasattr(s, 'experiment'):
            s = s.mean('experiment')
        if hasattr(s, 'atlas'):
            s = s.mean('atlas')
        if hasattr(s, 'layer'):
            max_score = s.sel(aggregation='center').max()
            s = s[{'layer': (s.sel(aggregation='center') == max_score).values}].squeeze('layer')
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
        ax.text(_x, _y, model, fontdict=dict(fontsize=10))

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

    _savefig(fig, f"{benchmark1}__{benchmark2}")


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
        data = pd.read_csv(store_file)
        data = data[data['model'].isin(models)]
        return data
    data = []
    for model in tqdm(models, desc='model scores'):
        model_scores = score(benchmark=benchmark, model=model)
        adjunct_columns = list(set(model_scores.dims) - {'aggregation'})
        for adjunct_values in itertools.product(*[model_scores[column].values for column in adjunct_columns]):
            adjunct_values = dict(zip(adjunct_columns, adjunct_values))
            current_score = model_scores.sel(**adjunct_values)
            data.append({**adjunct_values, **{
                'benchmark': benchmark, 'model': model,
                'score': current_score.sel(aggregation='center').values.tolist(),
                'error': current_score.sel(aggregation='error').values.tolist()}})
    data = pd.DataFrame(data)
    data.to_csv(store_file, index=False)
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


def fmri_brain_network_correlations():
    scores = collect_scores(benchmark='Pereira2018-encoding', models=models)
    # build correlation matrix
    correlations = np.zeros((len(fmri_atlases), len(fmri_atlases)))
    for i_x, i_y in itertools.combinations(list(range(len(fmri_atlases))), 2):
        benchmark_x, benchmark_y = fmri_atlases[i_x], fmri_atlases[i_y]
        x_data = scores[scores['atlas'] == benchmark_x]
        y_data = scores[scores['atlas'] == benchmark_y]
        x_data, y_data = align_both(x_data, y_data, on='model')
        x, xerr = x_data['score'].values.squeeze(), x_data['error'].values.squeeze()
        y, yerr = y_data['score'].values.squeeze(), y_data['error'].values.squeeze()
        r, p = pearsonr(x, y)
        significance_threshold = .05
        if p >= significance_threshold:
            r = 0
        correlations[i_x, i_y] = correlations[i_y, i_x] = r
    for i in range(len(fmri_atlases)):  # set diagonal to 1
        correlations[i, i] = 1

    # plot
    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.grid(False)
    ax.imshow(correlations, cmap=pyplot.get_cmap('Greens'), vmin=.85)
    for x, y in itertools.product(*[list(range(s)) for s in correlations.shape]):
        r = correlations[x, y]
        r = f"{r:.2f}" if r != 0 else 'n.s.'
        ax.text(x, y, r, ha='center', va='center', fontdict=dict(fontsize=10), color='white')
    # ticks
    ax.set_xticks(range(len(fmri_atlases)))
    ax.set_xticklabels(fmri_atlases, rotation=90)
    ax.set_yticks(range(len(fmri_atlases)))
    ax.set_yticklabels(fmri_atlases)
    ax.xaxis.tick_top()
    ax.tick_params(axis=u'both', which=u'both',
                   length=0)  # hide tick marks, but not text https://stackoverflow.com/a/29988431/2225200
    # save
    fig.tight_layout()
    _savefig(fig, 'brain_network_correlations')


def align_both(data1, data2, on):
    data1 = data1[data1[on].isin(data2[on])]
    data2 = data2[data2[on].isin(data1[on])]
    data1 = data1.set_index(on).reindex(index=data2[on]).reset_index()
    return data1, data2


def _savefig(fig, savename):
    fig.tight_layout()
    savepath = Path(__file__).parent / f"{savename}.png"
    logger.info(f"Saving to {savepath}")
    fig.savefig(savepath)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
