from collections import defaultdict

from numpy.random import RandomState
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

from neural_nlp.analyze import savefig, score_formatter, stats
from neural_nlp.analyze.scores import models as all_models, fmri_atlases, model_colors, \
    collect_scores, average_adjacent, choose_best_scores, collect_Pereira_experiment_scores, \
    align_scores, significance_stars, get_ceiling, shaded_errorbar, model_label_replace, \
    benchmark_label_replace, glue_benchmarks
from result_caching import is_iterable

_logger = logging.getLogger(__name__)


def retrieve_scores(benchmark, models=all_models):
    scores = collect_scores(benchmark, models)
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


def predictor(benchmark1, benchmarks2, ylim=None, num_bootstraps=1000, bootstrap_size=.9, color='#ababab'):
    # collect
    data = []
    for benchmark2 in benchmarks2:
        scores1 = retrieve_scores(benchmark1)
        scores2 = retrieve_scores(benchmark2)
        scores1, scores2 = align_scores(scores1, scores2, identifier_set=('model',))
        x = scores1['score'].values if benchmark1 != 'wikitext-2' else np.exp(scores1['score'].values)
        y = scores2['score'].values
        # compute overall p value
        _, p = pearsonr(x, y)
        # bootstrap error bars
        rng = RandomState(0)
        rs = []
        for bootstrap in range(num_bootstraps):
            indices = rng.choice(np.arange(len(x)), size=int(bootstrap_size * len(x)))
            bootstrap_x, bootstrap_y = x[indices], y[indices]
            r, _ = pearsonr(bootstrap_x, bootstrap_y)
            if benchmark1.startswith('wikitext'):
                r = -r  # flip because lower perplexity is better
            rs.append(r)
        data.append({'benchmark1': benchmark1, 'benchmark2': benchmark2, 'r': np.mean(rs), 'err': np.std(rs), 'p': p})
    data = pd.DataFrame(data)

    # plot
    _plot_predictor(benchmark1, data, benchmark_labels=[benchmark_label_replace[b] for b in benchmarks2],
                    color=color, ylim=ylim)


def untrained_predictor(benchmarks, ylim=None, num_bootstraps=1000, bootstrap_size=.9, color='#ababab'):
    # collect
    data = []
    for benchmark in benchmarks:
        models = [[model, f"{model}-untrained"] for model in all_models]
        models = [model for model_tuple in models for model in model_tuple]
        scores = retrieve_scores(benchmark, models=models)
        untrained_rows = np.array([model.endswith('-untrained') for model in scores['model']])
        scores1, scores2 = scores[~untrained_rows], scores[untrained_rows]

        x = scores1['score'].values
        y = scores2['score'].values
        # compute overall p value
        _, p = pearsonr(x, y)
        # bootstrap error bars
        rng = RandomState(0)
        rs = []
        for bootstrap in range(num_bootstraps):
            indices = rng.choice(np.arange(len(x)), size=int(bootstrap_size * len(x)))
            bootstrap_x, bootstrap_y = x[indices], y[indices]
            r, _ = pearsonr(bootstrap_x, bootstrap_y)
            rs.append(r)
        data.append({'benchmark': benchmark, 'r': np.mean(rs), 'err': np.std(rs), 'p': p})
    data = pd.DataFrame(data)

    # plot
    _plot_predictor(title='untrained', data=data, benchmark_labels=[benchmark_label_replace[b] for b in benchmarks],
                    color=color, ylim=ylim)


def _plot_predictor(title, data, benchmark_labels, color='#ababab', ylim=None):
    fig, ax = pyplot.subplots(figsize=(3, 4))
    width = 0.5
    x, y, yerr, p = data.index / 1.5, data['r'], data['err'], data['p']
    ax.bar(x, height=y, yerr=yerr, align='center', width=width, color=color, alpha=0.5, edgecolor='none',
           ecolor='gray', error_kw=dict(elinewidth=1, alpha=.5))
    for i, (xpos, ypos, pvalue) in enumerate(zip(x, y, p)):
        ax.text(x=xpos + .15 * width / 2, y=.01, s=benchmark_labels[i],
                rotation=90, rotation_mode='anchor',
                fontdict=dict(fontsize=20), color='black')  # 16
        ax.text(x=xpos, y=-.05, s=significance_stars(pvalue) if pvalue < .05 else 'n.s.', horizontalalignment='center',
                fontdict=dict(fontsize=14, fontweight='normal'))
    ax.set_ylabel('Correlation')
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(score_formatter)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y', which=u'both', length=0)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_linewidth(0.75)
    savefig(fig, Path(__file__).parent / f"predictor-{title}")


def task_predictors(target_benchmark='overall_neural-encoding',
                    num_bootstraps=1000, bootstrap_size=.9, bar_labels=False):
    # collect
    predictors = ['wikitext-2'] + glue_benchmarks
    correlation_data, values_data = [], defaultdict(list)
    for predictor in predictors:
        predictor_scores = retrieve_scores(predictor)
        scores = retrieve_scores(target_benchmark)
        predictor_scores, scores = align_scores(predictor_scores, scores, identifier_set=('model',))
        x = predictor_scores['score'] if not predictor.startswith('wikitext-2') else np.exp(predictor_scores['score'])
        y = scores['score']
        x, y = x.values, y.values
        values_data['x'].extend(x)
        values_data['y'].extend(y)
        values_data['predictor'].extend([predictor] * len(x))
        # compute overall p value
        _, p = pearsonr(x, y)
        # bootstrap error bars
        rng = RandomState(0)
        rs = []
        for bootstrap in range(num_bootstraps):
            indices = rng.choice(np.arange(len(x)), size=int(bootstrap_size * len(x)))
            bootstrap_x, bootstrap_y = x[indices], y[indices]
            r, _ = pearsonr(bootstrap_x, bootstrap_y)
            if predictor.startswith('wikitext'):
                r = -r  # flip because lower perplexity is better
            rs.append(r)
        correlation_data.append({'predictor': predictor, 'target': target_benchmark,
                                 'r': np.mean(rs), 'err': np.std(rs), 'p': p})
    correlation_data = pd.DataFrame(correlation_data)
    assert all(correlation_data['predictor'].values == predictors)  # ensure ordering

    # stats
    values_data = pd.DataFrame(values_data)
    interactions = stats.interaction_test(values_data, category_column='predictor', compare_only='wikitext-2')
    pd.options.display.width = 0
    print(interactions)

    # plot
    groups = [[0], [1], [2], [3, 4, 5], [6, 7], [8]]  # group GLUE benchmarks
    fig, ax = pyplot.subplots(figsize=(5, 6))
    wiki_color = '#0035ff'
    width = 0.5
    y, yerr, p = correlation_data['r'], correlation_data['err'], correlation_data['p']
    x, color, group_color, i = [], [], [], 0
    for group_iter, group in enumerate(groups):
        mono_color = 0.2 + (group_iter / len(groups)) * (1.0 - 0.2)
        group_color.append((mono_color, mono_color, mono_color))
        for _ in group:
            x.append(i)
            color.append((mono_color, mono_color, mono_color))
            i += 1.1 * width
        i += width / 2
    color[0] = group_color[0] = wiki_color
    ax.bar(x, height=y, yerr=yerr, align='center', width=width,
           color=color, edgecolor='none', ecolor='gray', error_kw=dict(elinewidth=1, alpha=.5))
    # legend
    labels = ['next-word prediction', 'sentence grammaticality (CoLA)', 'sentence sentiment (SST-2)',
              'semantic similarity (QQP, MRPC, STS-B)', 'entailment (MNLT, RTE)', 'question-answer coherence (QNLI)']
    handles = [pyplot.Rectangle((0, 0), 1, 1, color=color) for color in group_color]
    legend = ax.legend(handles, labels, prop=dict(size=9))
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor('black')
    # text
    for i, (xpos, ypos, pvalue) in enumerate(zip(x, y, p)):
        if bar_labels:
            ax.text(x=xpos + .5 * width / 2, y=.01, s=benchmark_label_replace[predictors[i]],
                    rotation=90, rotation_mode='anchor',
                    fontdict=dict(fontsize=12), color='white' if i == 0 else 'black')
        ax.text(x=xpos, y=-.03, s=significance_stars(pvalue) if pvalue < .05 else 'n.s.', horizontalalignment='center',
                fontdict=dict(fontsize=10, fontweight='bold' if i == 0 else 'normal'))
    # plot formatting
    ax.set_ylabel('Correlation')
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(score_formatter)
    ax.set_ylim([-.5, .6])
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_linewidth(0.75)
    savefig(fig, Path(__file__).parent / "task_predictors")


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
