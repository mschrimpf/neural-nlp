import fire
import numpy as np
from matplotlib import pyplot
from pathlib import Path

from neural_nlp.analyze.scores import collect_scores, models as all_models, fmri_atlases
import seaborn


def layer_preference_per_region(models=None):
    models = models or [model for model in all_models if model != 'glove']  # glove has only 1 layer
    data = collect_scores(benchmark='Pereira2018-encoding', models=models)
    data = data.groupby(['benchmark', 'model', 'atlas', 'layer'], sort=False)[['score', 'error']].mean().reset_index()

    assert set(data['atlas']) == set(fmri_atlases)
    fig, axes = pyplot.subplots(figsize=(20, 6), ncols=len(fmri_atlases))
    for i, (ax, atlas) in enumerate(zip(axes, fmri_atlases)):
        ax.set_title(atlas)
        atlas_data = data[data['atlas'] == atlas]
        for model, group in atlas_data.groupby('model'):
            num_layers = len(group['layer'])  # assume layers are correctly ordered
            relative_position = np.arange(num_layers) / (num_layers - 1)
            shaded_errorbar(x=relative_position, y=group['score'], error=group['error'], label=model, ax=ax,
                            alpha=0.4, shaded_kwargs=dict(alpha=0.2))
        ax.set_ylim([-.02, .33])
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('score')
    # xlabel
    fig.text(0.5, 0.01, 'relative layer position', ha='center')
    # legend
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.86), loc='center', ncol=len(labels))
    for legend_handle in legend.legendHandles:
        legend_handle.set_alpha(1)
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
