import fire
import numpy as np
from matplotlib import pyplot
from pathlib import Path

from neural_nlp.analyze.scores import collect_scores, models as all_models, fmri_atlases, model_colors
import seaborn

from neural_nlp.analyze.scores.bars import model_ordering


def layer_preference_per_region(models=None):
    models = models or [model for model in all_models if model != 'glove']  # glove has only 1 layer
    models = model_ordering(models, benchmark='Pereira2018-encoding')  # order by best scores
    data = collect_scores(benchmark='Pereira2018-encoding', models=models)
    data = data.groupby(['benchmark', 'model', 'atlas', 'layer'], sort=False)[['score', 'error']].mean().reset_index()

    models_group1 = ['lm_1b', 'openaigpt', 'gpt2', 'gpt2-medium', 'gpt2-large']
    models_group2 = [model for model in models if model not in models_group1]
    model_groups = [models_group1, models_group2]
    ylims = [0.4, 0.3]
    assert set(data['atlas']) == set(fmri_atlases)
    fig, axes = pyplot.subplots(figsize=(20, 12), nrows=2, ncols=len(fmri_atlases))
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
                            bbox_to_anchor=(0.5, -.05 + 0.49 * (len(model_groups) - model_group_iter)), loc='center')
        for legend_handle in legend.legendHandles:
            legend_handle.set_alpha(1)
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # save
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / 'layer_ordering.png', dpi=600)


def shaded_errorbar(x, y, error, ax=None, shaded_kwargs=None, **kwargs):
    shaded_kwargs = shaded_kwargs or {}
    ax = ax or pyplot.gca()
    line = ax.plot(x, y, **kwargs)
    ax.fill_between(x, y - error, y + error, **shaded_kwargs)
    return line


if __name__ == '__main__':
    seaborn.set(context='talk')
    fire.Fire()
