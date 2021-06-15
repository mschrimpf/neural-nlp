import copy
from collections import OrderedDict

import fire
import logging
import matplotlib
import numpy as np
import seaborn
import sys
from matplotlib import pyplot
from pathlib import Path
from scipy.signal import savgol_filter

from neural_nlp import model_layers
from neural_nlp.analyze import savefig
from neural_nlp.analyze.scores import collect_scores, models as all_models, model_colors, \
    fmri_atlases, shaded_errorbar, average_adjacent, benchmark_label_replace, model_label_replace

_logger = logging.getLogger(__name__)

model_groups = OrderedDict([
    # BERT
    ('BERT', ['bert-base-uncased', 'bert-base-multilingual-cased', 'bert-large-uncased',
              'bert-large-uncased-whole-word-masking', 'distilbert-base-uncased']),
    # GPT
    ('GPT', ['distilgpt2', 'openaigpt', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']),
    # Transfo-XL + XLNet
    ('Transfo-XL/XLNet', ['transfo-xl-wt103', 'xlnet-base-cased', 'xlnet-large-cased']),
    # XLM
    ('XLM', ['xlm-mlm-en-2048', 'xlm-mlm-enfr-1024', 'xlm-mlm-xnli15-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280']),
    # RoBERTa
    ('RoBERTa', ['roberta-base', 'roberta-large', 'distilroberta-base', 'xlm-roberta-base', 'xlm-roberta-large']),
    # AlBERT
    ('AlBERT', ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2',
                'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2']),
    # T5
    ('T5', ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']),
])


def layer_preference_per_region(models=None):
    models = models or [model for model in all_models if len(model_layers[model]) > 1]  # need at least 2 layers to plot
    data = collect_scores(benchmark='Pereira2018-encoding', models=models)
    data = data.groupby(['benchmark', 'model', 'atlas', 'layer'], sort=False)[['score', 'error']].mean().reset_index()

    groups = list(model_groups.values())
    groups.append([model for model in models if not any(model in group for group in groups)])
    ylims = [0.35] * 8
    assert set(data['atlas']) == set(fmri_atlases)
    fig, axes = pyplot.subplots(figsize=(20, 6 * len(groups)), nrows=len(groups), ncols=len(fmri_atlases))
    for model_group_iter, (models, ylim) in enumerate(zip(groups, ylims)):
        for atlas_iter, atlas in enumerate(fmri_atlases):
            ax = axes[model_group_iter, atlas_iter]
            ax.set_title(atlas)
            atlas_data = data[data['atlas'] == atlas]
            for model in models:
                group = atlas_data[atlas_data['model'] == model]
                num_layers = len(group['layer'])  # assume layers are correctly ordered
                relative_position = np.arange(num_layers) / (num_layers - 1)
                shaded_errorbar(x=relative_position, y=group['score'], error=group['error'], label=model, ax=ax,
                                alpha=0.4, color=model_colors[model],
                                shaded_kwargs=dict(alpha=0.2, color=model_colors[model]))
            ax.set_ylim([-.02, ylim])
            if atlas_iter > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('score')
        # legend
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=len(labels),
                            bbox_to_anchor=(0.5, -0.025 + 0.125 * (len(groups) - model_group_iter)), loc='center')
        for legend_handle in legend.legendHandles:
            legend_handle.set_alpha(1)
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # save
    fig.tight_layout()
    savefig(fig, Path(__file__).parent / 'layer_ordering')


def layer_preference(benchmark='Pereira2018-encoding'):
    models = [model for model in all_models if len(model_layers[model]) > 1]  # need at least 2 layers to plot
    data = collect_scores(benchmark=benchmark, models=models)
    data = average_adjacent(data)

    groups = copy.deepcopy(model_groups)
    groups['other'] = [model for model in models if not any(model in group for group in groups.values())]
    _logger.debug(f"Non-assigned models: {groups['other']}")
    fig, axes = pyplot.subplots(figsize=(20, 6), nrows=1, ncols=len(groups), sharey=True)
    for model_group_iter, (ax, (group_name, models)) in enumerate(zip(axes.flatten(), groups.items())):
        ax.set_title(group_name)
        for model in models:
            group = data[data['model'] == model]
            num_layers = len(group['layer'])  # assume layers are correctly ordered
            relative_position = np.arange(num_layers) / (num_layers - 1)
            shaded_errorbar(x=relative_position, y=group['score'], error=group['error'], label=model, ax=ax,
                            alpha=0.4, color=model_colors[model],
                            shaded_kwargs=dict(alpha=0.2, color=model_colors[model]))
        if model_group_iter > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('score')
        ax.set_ylim([0, 1.2])
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # save
    fig.tight_layout()
    savefig(fig, Path(__file__).parent / f'layer_ordering-{benchmark}')


def layer_preference_single(model='gpt2-xl',
                            benchmarks=('Pereira2018-encoding', 'Fedorenko2016v3-encoding', 'Blank2014fROI-encoding'),
                            smoothing=False,
                            ):
    data = [collect_scores(benchmark=benchmark, models=[model]) for benchmark in benchmarks]
    data = [average_adjacent(d) for d in data]

    fig, axes = pyplot.subplots(figsize=(15, 6), nrows=1, ncols=len(benchmarks), sharey=True)
    axes = axes.flatten() if len(benchmarks) > 1 else [axes]
    for benchmark_iter, (ax, benchmark_name, benchmark_data) in enumerate(zip(axes, benchmarks, data)):
        ax.set_title(benchmark_label_replace[benchmark_name])
        num_layers = len(benchmark_data['layer'])  # assume layers are correctly ordered
        relative_position = np.arange(num_layers) / (num_layers - 1)
        y, error = benchmark_data['score'], benchmark_data['error']
        if smoothing:
            window_size = 11
            y = savgol_filter(y, window_size, 3)
        shaded_errorbar(x=relative_position, y=y, error=error, label=model, ax=ax,
                        alpha=0.4, color=model_colors[model],
                        linewidth=7.0 if model == 'gpt2-xl' else 1.0,
                        shaded_kwargs=dict(alpha=0.2, color=model_colors[model]))
        if benchmark_iter == 0:
            ax.set_ylabel('Normalized Predictivity')
        ax.set_ylim([0, 1.1])
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # save
    fig.tight_layout()
    savefig(fig, Path(__file__).parent / f'layer_ordering-{model}')


def layer_training_delta(models=None):
    models = models or [model for model in all_models if len(model_layers[model]) > 1]  # need at least 2 layers to plot
    scores_models = [(model, f"{model}-untrained") for model in models]
    scores_models = [model for model_tuple in scores_models for model in model_tuple]
    data = collect_scores(benchmark='Pereira2018-encoding', models=scores_models)
    data = data.groupby(['benchmark', 'model', 'atlas', 'layer'], sort=False)[['score', 'error']].mean().reset_index()

    model_groups.append([model for model in models if not any(model.rstrip() in group for group in model_groups)])
    ylims = [0.35] * 8
    assert set(data['atlas']) == set(fmri_atlases)
    fig, axes = pyplot.subplots(figsize=(20, 6 * len(model_groups)), nrows=len(model_groups), ncols=len(fmri_atlases))
    for model_group_iter, (models, ylim) in enumerate(zip(model_groups, ylims)):
        for atlas_iter, atlas in enumerate(fmri_atlases):
            ax = axes[model_group_iter, atlas_iter]
            ax.set_title(atlas)
            atlas_data = data[data['atlas'] == atlas]
            for model in models:
                trained_scores = atlas_data[atlas_data['model'] == model]
                untrained_scores = atlas_data[atlas_data['model'] == f"{model}-untrained"]
                if len(untrained_scores) == 0:
                    continue
                num_layers = len(trained_scores['layer'])  # assume layers are correctly ordered
                relative_position = np.arange(num_layers) / (num_layers - 1)
                y = trained_scores['score'].values - untrained_scores['score'].values
                error = np.maximum.reduce([trained_scores['error'], untrained_scores['error']])
                shaded_errorbar(x=relative_position, y=y, error=error, label=model, ax=ax,
                                alpha=0.4, color=model_colors[model],
                                shaded_kwargs=dict(alpha=0.2, color=model_colors[model]))
            ax.set_ylim([-.02, ylim])
            if atlas_iter > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('score')
        # legend
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, ncol=len(labels),
                            # bbox_to_anchor=(0.5, -.05 + 0.49 * (len(model_groups) - model_group_iter)), loc='center')
                            bbox_to_anchor=(0.5, -0.025 + 0.125 * (len(model_groups) - model_group_iter)), loc='center')
        for legend_handle in legend.legendHandles:
            legend_handle.set_alpha(1)
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # save
    fig.tight_layout()
    savefig(fig, Path(__file__).parent / 'layer_ordering-deltas')


def first_last_layer_scores(benchmarks=('Pereira2018-encoding', 'Fedorenko2016v3-encoding', 'Blank2014fROI-encoding')):
    models = all_models
    data = [collect_scores(benchmark=benchmark, models=models) for benchmark in benchmarks]
    data = [average_adjacent(d) for d in data]

    @matplotlib.ticker.FuncFormatter
    def score_formatter(score, pos):
        if score < 0 or score > 1:
            return ""
        return f"{score:.2f}"

    fig, axes = pyplot.subplots(figsize=(15, 15), nrows=len(benchmarks), ncols=1, sharey=False)
    width = 0.5
    for benchmark_iter, (benchmark, benchmark_data) in enumerate(zip(benchmarks, data)):
        ax = axes[benchmark_iter]
        for model_iter, model in enumerate(models):
            model_data = benchmark_data[benchmark_data['model'] == model]
            best_score = model_data['score'].max()
            best_score_error = model_data[model_data['score'] == best_score]['error']
            ax.errorbar(x=model_iter, y=best_score, yerr=best_score_error,
                        marker='.', color='black',
                        label='best layer' if model_iter == len(models) - 1 else None)
            if len(model_data) > 1:
                first_score, first_score_error = model_data['score'].values[0], model_data['error'].values[0]
                ax.errorbar(x=model_iter - 0.2 * width, y=first_score, yerr=first_score_error,
                            marker='.', color='lightgray',
                            label='first layer' if model_iter == len(models) - 1 else None)
                last_score, last_score_error = model_data['score'].values[-1], model_data['error'].values[-1]
                ax.errorbar(x=model_iter + 0.2 * width, y=last_score, yerr=last_score_error,
                            marker='.', color='gray',
                            label='last layer' if model_iter == len(models) - 1 else None)
        if benchmark_iter < len(benchmarks) - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels([model_label_replace[model] for model in models], rotation=90)
        if benchmark_iter == 0:
            ax.legend()
        ax.set_ylabel('Normalized Predictivity')
        ax.set_title(benchmark_label_replace[benchmark])
        ax.yaxis.set_major_formatter(score_formatter)
    savefig(fig, Path(__file__).parent / 'first_layer_layer_scores')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for shush_logger in ['matplotlib']:
        logging.getLogger(shush_logger).setLevel(logging.WARNING)
    seaborn.set(context='talk')
    fire.Fire()
