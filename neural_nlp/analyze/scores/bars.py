import fire
import itertools
import logging
import numpy as np
import pandas as pd
import seaborn
import sys
from decimal import Decimal
from functools import reduce
from matplotlib import pyplot, patheffects
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.stats import pearsonr

from neural_nlp.analyze import savefig, score_formatter
from neural_nlp.analyze.scores import models as all_models, fmri_atlases, model_colors, \
    collect_scores, average_adjacent, choose_best_scores, collect_Pereira_experiment_scores, \
    align_scores, significance_stars, get_ceiling, shaded_errorbar, model_label_replace, \
    benchmark_label_replace
from result_caching import is_iterable

_logger = logging.getLogger(__name__)


def retrieve_scores(benchmark):
    scores = collect_scores(benchmark, all_models)
    scores = average_adjacent(scores)  # average each model+layer's score per experiment and atlas
    scores = scores.fillna(0)  # nan scores are 0
    scores = choose_best_scores(scores)
    nan = scores[scores.isna().any(1)]
    if len(nan) > 0:
        _logger.warning(f"Dropping nan rows: {nan}")
        scores = scores.dropna()
    return scores


def fmri_per_network(models=all_models, benchmark='Pereira2018-encoding'):
    data = collect_scores(benchmark, models)
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
    savefig(fig, Path(__file__).parent / f'bars-network-{benchmark}')


def wikitext_best(benchmark='wikitext-2'):
    whole_best(benchmark=benchmark, ylabel='NLL / Perplexity', ylim=50)


def whole_best(benchmark=None, data=None, title_kwargs=None, normalize_error=False, **kwargs):
    data = data if data is not None else retrieve_scores(benchmark)
    models = [model for model in all_models if model in data['model'].values]
    fig, ax = pyplot.subplots(figsize=(5, 4))
    ax.set_title(benchmark_label_replace[benchmark], **(title_kwargs or {}))
    ceiling, ceiling_err = get_ceiling(benchmark, which='both', normalize_scale=normalize_error)
    _plot_bars(ax, models=models, data=data, text_kwargs=dict(fontdict=dict(fontsize=6)), **kwargs)
    if is_iterable(ceiling_err) or not np.isnan(ceiling_err):  # no performance benchmarks
        ceiling_y = 1  # we already normalized so ceiling == 1
        xlim = ax.get_xlim()
        ax.plot([-50, +50], [ceiling_y, ceiling_y], color='gray')
        shaded_errorbar(x=[-50, +50], y=np.array([ceiling_y, ceiling_y]),
                        error=([ceiling_err[0], ceiling_err[0]], [ceiling_err[1], ceiling_err[1]]),
                        ax=ax, alpha=0, shaded_kwargs=dict(color='darkgray', alpha=.5))
        ax.set_xlim(xlim)
        ax.set_ylim([-.05, max(1.05 + ceiling_err[-1], max(data['score']) + .05)])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
    ax.yaxis.set_major_formatter(score_formatter)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_linewidth(0.75)
    savefig(fig, Path(__file__).parent / f'bars-{benchmark}')


def _plot_bars(ax, models, data, ylim=None, width=0.5, ylabel="Normalized Predictivity", annotate=True,
               text_kwargs=None):
    text_kwargs = {**dict(fontdict=dict(fontsize=7), color='white'), **(text_kwargs or {})}
    step = (len(models) + 1) * width
    offset = len(models) / 2
    for model_iter, model in enumerate(models):
        model_score = data[data['model'] == model]
        y, yerr = model_score['score'], model_score['error']
        x = np.arange(start=0, stop=len(y) * step, step=step)
        model_x = x - offset * width + model_iter * width
        ax.bar(model_x, height=y, yerr=yerr, width=width,
               edgecolor='none', color=model_colors[model], ecolor='gray', error_kw=dict(elinewidth=1, alpha=.5))
        if annotate:
            for xpos in model_x:
                ax.text(x=xpos + .6 * width / 2, y=.005, s=model_label_replace[model],
                        rotation=90, rotation_mode='anchor', **text_kwargs)
    for (model_group, annotate_x, width) in [
        ('Emb.', 0.077, 1.05),
        ('BERT', 0.203, 1.76),
        ('XLM', 0.3935, 2.44),
        ('T5', 0.605, 1.75),
        ('AlBERT', 0.7427, 2.78),
        ('GPT', 0.8904, 2.085),
    ]:
        ax.annotate(model_group, xy=(annotate_x, +.0), xytext=(annotate_x, -.05), xycoords='axes fraction',
                    fontsize=8, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=.3', lw=1.0, color='black'))

    ax.set_ylabel(ylabel, fontdict=dict(fontsize=16))
    if ylim is not None and ylim <= 1:
        ax.set_yticks(np.arange(0, ylim, .1))
        ax.set_yticklabels([
            "0" if Decimal(f"{label:.2f}") == Decimal('0') else "1" if Decimal(f"{label:.2f}") == Decimal('1') else
            f"{label:.1f}"[1:] if Decimal(f"{label:.2f}") % Decimal(".2") == 0 else ""
            for label in ax.get_yticks()], fontdict=dict(fontsize=14))
    ax.set_xticks(x)
    ax.tick_params(axis="x", pad=-5)
    return ax


def random_embedding():
    models = ['gpt2-xl', 'gpt2-xl-untrained', 'random-embedding']
    benchmarks = ['Pereira2018-encoding', 'Fedorenko2016v3-encoding', 'Blank2014fROI-encoding', 'Futrell2018-encoding']
    scores = [collect_scores(benchmark=benchmark, models=models) for benchmark in benchmarks]
    scores = [average_adjacent(benchmark_scores) for benchmark_scores in scores]
    scores = [choose_best_scores(benchmark_scores).dropna() for benchmark_scores in scores]
    scores = reduce(lambda left, right: pd.concat([left, right]), scores)

    fig, ax = pyplot.subplots(figsize=(5, 4))
    colors = {'gpt2-xl': model_colors['gpt2-xl'], 'gpt2-xl-untrained': '#284343', 'random-embedding': '#C3CCCC'}
    offsets = {0: -.2, 1: 0, 2: +.2}
    width = 0.5 / 3
    text_kwargs = dict(fontdict=dict(fontsize=7), color='white')
    base_x = np.arange(len(benchmarks))
    for i, model in enumerate(models):
        model_scores = scores[scores['model'] == model]
        x = base_x + offsets[i]
        ax.bar(x, height=model_scores['score'], yerr=model_scores['error'],
               width=width, align='center',
               color=colors[model], edgecolor='none', ecolor='gray', error_kw=dict(elinewidth=1, alpha=.5))
        for xpos in x:
            text = ax.text(x=xpos + .6 * width / 2, y=.05, s=model_label_replace[model],
                           rotation=90, rotation_mode='anchor', **text_kwargs)
            text.set_path_effects([patheffects.withStroke(linewidth=0.75, foreground='black')])
    ax.set_xticks(base_x)
    ax.set_xticklabels([benchmark_label_replace[benchmark] for benchmark in benchmarks], fontsize=9)
    ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
    ax.yaxis.set_major_formatter(score_formatter)
    ax.set_ylim([0, 1.2])
    ax.set_ylabel('Normalized Predictivity')
    savefig(fig, Path(__file__).parent / "bars-random_embedding")


def shortcomings(model):
    benchmarks = ['Futrell2018-encoding', 'Futrell2018sentences-encoding', 'Futrell2018stories-encoding']
    scores = [collect_scores(models=[model], benchmark=benchmark) for benchmark in benchmarks]
    scores = reduce(lambda left, right: pd.concat([left, right]), scores)
    scores = average_adjacent(scores).dropna()
    scores = choose_best_scores(scores)
    fig, ax = pyplot.subplots(figsize=(3.5, 5))
    x = np.arange(len(benchmarks))
    ax.bar(x, height=scores['score'], yerr=scores['error'],
           color=model_colors[model], edgecolor='none', ecolor='gray', error_kw=dict(elinewidth=1, alpha=.5))
    ax.set_xticks(x)
    ax.set_xticklabels(['words', 'sentences', 'stories'], rotation=45)
    ax.yaxis.set_major_locator(MultipleLocator(base=0.2))
    ax.yaxis.set_major_formatter(score_formatter)
    ax.set_ylim([0, 1.2])
    ax.set_ylabel('Normalized Predictivity')
    ax.set_title('Futrell2018 variations')
    savefig(fig, Path(__file__).parent / "bars-generalization")


def benchmark_correlations(best_layer=True):
    data = []
    # Pereira internal
    Pereira_experiment2_scores, Pereira_experiment3_scores = collect_Pereira_experiment_scores(best_layer=best_layer)
    Pereira_experiment2_scores = Pereira_experiment2_scores['score'].values
    Pereira_experiment3_scores = Pereira_experiment3_scores['score'].values
    correlation_Pereira, p_Pereira = pearsonr(Pereira_experiment2_scores, Pereira_experiment3_scores)
    data.append(dict(benchmark1='Pereira Exp. 2', benchmark2='Pereira Exp. 3', r=correlation_Pereira, p=p_Pereira))
    # cross-benchmark
    benchmarks = ('Pereira2018-encoding', 'Blank2014fROI-encoding', 'Fedorenko2016-encoding')
    for benchmark1, benchmark2 in itertools.combinations(benchmarks, 2):
        benchmark1_scores = collect_scores(benchmark=benchmark1, models=all_models)
        benchmark2_scores = collect_scores(benchmark=benchmark2, models=all_models)
        benchmark1_scores = average_adjacent(benchmark1_scores).dropna()
        benchmark2_scores = average_adjacent(benchmark2_scores).dropna()
        if best_layer:
            benchmark1_scores = choose_best_scores(benchmark1_scores)
            benchmark2_scores = choose_best_scores(benchmark2_scores)
        benchmark1_scores, benchmark2_scores = align_scores(
            benchmark1_scores, benchmark2_scores, identifier_set=('model',) if best_layer else ('model', 'layer'))
        benchmark1_scores, benchmark2_scores = benchmark1_scores['score'].values, benchmark2_scores['score'].values
        r, p = pearsonr(benchmark1_scores, benchmark2_scores)
        data.append(dict(benchmark1=benchmark1, benchmark2=benchmark2, r=r, p=p))
    data = pd.DataFrame(data)
    # plot
    fig, ax = pyplot.subplots(figsize=(3, 4))
    x = np.arange(len(data))
    ax.bar(x, data['r'])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{benchmark1[:5]} / {benchmark2[:5]}" for benchmark1, benchmark2 in
                        zip(data['benchmark1'].values, data['benchmark2'].values)], rotation=90)
    for _x, r, p in zip(x, data['r'].values, data['p'].values):
        ax.text(_x, r + .05, significance_stars(p) if p < .05 else 'n.s.', fontsize=12,
                horizontalalignment='center', verticalalignment='center')
    savefig(fig, Path(__file__).parent / 'benchmark-correlations' + ('-best' if best_layer else '-layers'))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
