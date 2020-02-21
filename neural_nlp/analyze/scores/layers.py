import fire
import numpy as np
import seaborn
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import model_layers
from neural_nlp.analyze.scores import collect_scores, models as all_models, fmri_atlases, model_colors, shaded_errorbar

model_groups = [
    # BERT
    ['bert-base-uncased', 'bert-base-multilingual-cased', 'bert-large-uncased',
     'bert-large-uncased-whole-word-masking', 'distilbert-base-uncased'],
    # GPT
    ['openaigpt', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    # Transfo-XL + XLNet
    ['transfo-xl-wt103', 'xlnet-base-cased', 'xlnet-large-cased'],
    # XLM
    ['xlm-mlm-en-2048', 'xlm-mlm-enfr-1024', 'xlm-mlm-xnli15-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280'],
    # RoBERTa
    ['roberta-base', 'roberta-large', 'distilroberta-base'],
    # AlBERT
    ['albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2',
     'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2'],
    # T5
    ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'],
]


def layer_preference_per_region(models=None):
    models = models or [model for model in all_models if len(model_layers[model]) > 1]  # need at least 2 layers to plot
    data = collect_scores(benchmark='Pereira2018-encoding', models=models)
    data = data.groupby(['benchmark', 'model', 'atlas', 'layer'], sort=False)[['score', 'error']].mean().reset_index()

    model_groups.append([model for model in models if not any(model in group for group in model_groups)])
    ylims = [0.35] * 8
    assert set(data['atlas']) == set(fmri_atlases)
    fig, axes = pyplot.subplots(figsize=(20, 6 * len(model_groups)), nrows=len(model_groups), ncols=len(fmri_atlases))
    for model_group_iter, (models, ylim) in enumerate(zip(model_groups, ylims)):
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
                            # bbox_to_anchor=(0.5, -.05 + 0.49 * (len(model_groups) - model_group_iter)), loc='center')
                            bbox_to_anchor=(0.5, -0.025 + 0.125 * (len(model_groups) - model_group_iter)), loc='center')
        for legend_handle in legend.legendHandles:
            legend_handle.set_alpha(1)
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # save
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / 'layer_ordering.png', dpi=600)


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
    fig.savefig(Path(__file__).parent / 'layer_ordering-deltas.png', dpi=600)


if __name__ == '__main__':
    seaborn.set(context='talk')
    fire.Fire()
