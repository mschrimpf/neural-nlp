from decimal import Decimal

import fire
import logging
import numpy as np
import pandas as pd
import seaborn
import sys
from functools import reduce
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import benchmark_pool
from neural_nlp.analyze.scores import models as all_models, fmri_atlases, collect_scores, model_colors
from neural_nlp.benchmarks.neural import ceiling_normalize_score_error


def retrieve_scores(benchmark, normalize=False):
    if benchmark == 'Pereira2018-encoding':
        scores = collect_best_scores('Pereira2018-encoding', all_models)
        scores = average_experiments_atlases(scores)
    else:
        scores = collect_best_scores(benchmark, all_models)

    if normalize:
        benchmark = benchmark_pool[benchmark]()
        ceiling = benchmark.ceiling.sel(aggregation='center').values

        def apply(row):
            row['score'], row['error'] = ceiling_normalize_score_error(row['score'], row['error'], ceiling=ceiling)
            return row

        scores = scores.apply(apply, axis=1)
    return scores


def average_experiments_atlases(data):
    assert set(data['atlas']) == set(fmri_atlases)
    data = data.groupby(['benchmark', 'model', 'layer']).mean()  # mean across experiments & atlases
    return data.reset_index()


def fmri_per_network(models=all_models, benchmark='Pereira2018-encoding'):
    data = collect_best_scores(benchmark, models)
    ylim = 0.40
    models = [model for model in models if model in data['model'].values]
    assert set(data['atlas'].values) == set(fmri_atlases)
    data['atlas_order'] = [fmri_atlases.index(atlas) for atlas in data['atlas']]  # ensure atlases are same order
    data = data.sort_values('atlas_order').drop('atlas_order', axis=1)
    experiments = ('384sentences', '243sentences')
    assert set(data['experiment'].values) == set(experiments)
    # plot
    fig, axes = pyplot.subplots(figsize=(18, 6), nrows=2)
    for ax_iter, (experiment, ax) in enumerate(zip(experiments, axes)):
        ax.set_xlabel(experiment, fontdict=dict(fontsize=20))  # "title" at the bottom
        experiment_scores = data[data['experiment'] == experiment]

        _plot_bars(ax, models=models, data=experiment_scores, ylim=ylim)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', which='both', length=0)  # hide ticks
        ax.tick_params(axis='x', pad=3)  # shift labels
        if ax_iter == 0:
            ax.set_xticklabels(fmri_atlases, fontdict=dict(fontsize=12))
        else:
            ax.set_xticklabels([])

    fig.subplots_adjust(wspace=0, hspace=.2)
    pyplot.savefig(Path(__file__).parent / f'bars-network-{benchmark}.png', dpi=600)


def fmri_best(benchmark='Pereira2018-encoding'):
    whole_best(benchmark=benchmark, title=benchmark, ylim=.40)


def stories_best(benchmark='stories_froi_bold4s-encoding'):
    whole_best(benchmark=benchmark, title='stories fROI', ylim=.15)


def ecog_best(benchmark='Fedorenko2016-encoding'):
    whole_best(benchmark=benchmark, title='ECoG', ylim=.30)


def wikitext_best(benchmark='wikitext-2'):
    whole_best(benchmark=benchmark, title='wikitext-2', ylabel='NLL / Perplexity', ylim=50)


def overall(benchmarks=('Pereira2018-encoding', 'stories_froi_bold4s-encoding')):
    data = [retrieve_scores(benchmark, normalize=True) for benchmark in benchmarks]
    data = reduce(lambda left, right: pd.concat([left, right]), data)
    # note that this discards layer info and allows for different layers for different benchmarks which is not ideal.
    # A better way would be to at least select the layer based on its max average predictivity.
    data = data.groupby(['model']).mean().reset_index()
    whole_best(title=f"mean({', '.join(benchmarks)})", data=data, ylim=1., benchmark='average',
               ylabel='Normalized Predictivity (r/c)', title_kwargs=dict(fontdict=dict(fontsize=10)))


def whole_best(title, benchmark=None, data=None, title_kwargs=None, **kwargs):
    data = data if data is not None else retrieve_scores(benchmark)
    models = [model for model in all_models if model in data['model'].values]
    fig, ax = pyplot.subplots(figsize=(5, 4))
    ax.set_title(title, **(title_kwargs or {}))
    _plot_bars(ax, models=models, data=data, text_kwargs=dict(fontdict=dict(fontsize=9)), **kwargs)
    ax.set_xticks([])
    ax.set_xticklabels([])
    fig.tight_layout()
    pyplot.savefig(Path(__file__).parent / f'bars-{benchmark}.png', dpi=600)


def model_ordering(models, benchmark):
    fmri_data = collect_best_scores(benchmark=benchmark, models=models)
    mean_scores = fmri_data.groupby('model')['score'].mean()
    models = mean_scores.sort_values().index.values
    return models


def _plot_bars(ax, models, data, ylim, width=0.5, ylabel="Predictivity (Pearson r)", text_kwargs=None):
    text_kwargs = {**dict(fontdict=dict(fontsize=7), color='white'), **(text_kwargs or {})}
    step = (len(models) + 1) * width
    offset = len(models) / 2
    for model_iter, model in enumerate(models):
        model_score = data[data['model'] == model]
        y, yerr = model_score['score'], model_score['error']
        x = np.arange(start=0, stop=len(y) * step, step=step)
        model_x = x - offset * width + model_iter * width
        ax.bar(model_x, height=y, yerr=yerr, width=width, edgecolor='none', color=model_colors[model])
        for xpos in model_x:
            ax.text(x=xpos + .8 * width / 2, y=.01, s=model,
                    rotation=90, rotation_mode='anchor', **text_kwargs)
    ax.set_ylabel(ylabel, fontdict=dict(fontsize=10))
    ax.set_ylim([-.05, ylim])
    if ylim <= 1:
        ax.set_yticks(np.arange(0, ylim, .1))
        ax.set_yticklabels([
            "0" if Decimal(f"{label:.2f}") == Decimal('0') else "1" if Decimal(f"{label:.2f}") == Decimal('1') else
            f"{label:.1f}"[1:] if Decimal(f"{label:.2f}") % Decimal(".2") == 0 else ""
            for label in ax.get_yticks()], fontdict=dict(fontsize=14))
    ax.set_xticks(x)
    ax.tick_params(axis="x", pad=-5)
    return ax


def collect_best_scores(benchmark, models):
    scores = collect_scores(benchmark=benchmark, models=models)
    adjunct_columns = list(set(scores.columns) - {'score', 'error', 'layer'})
    scores = scores.loc[scores.groupby(adjunct_columns)['score'].idxmax()]  # max layer
    return scores


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
