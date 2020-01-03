import fire
import logging
import numpy as np
import seaborn
import sys
from matplotlib import pyplot
from pathlib import Path

from neural_nlp.analyze.scores import models as all_models, fmri_atlases, collect_scores, model_colors


def fmri_best(models=all_models, benchmark='Pereira2018-encoding'):
    data = collect_best_scores(benchmark, models)
    ylim = 0.40
    # ordering
    mean_scores = data.groupby('model')['score'].mean()
    models = mean_scores.sort_values().index.values
    assert set(data['atlas'].values) == set(fmri_atlases)
    data['atlas_order'] = [fmri_atlases.index(atlas) for atlas in data['atlas']]  # ensure atlases are same order
    data = data.sort_values('atlas_order').drop('atlas_order', axis=1)
    experiments = ('384sentences', '243sentences')
    assert set(data['experiment'].values) == set(experiments)
    # plot
    fig, axes = pyplot.subplots(nrows=2)
    for ax_iter, (experiment, ax) in enumerate(zip(experiments, axes)):
        ax.set_xlabel(experiment, fontdict=dict(fontsize=20))  # "title" at the bottom
        experiment_scores = data[data['experiment'] == experiment]

        def get_model_score(model):
            model_scores = experiment_scores[experiment_scores['model'] == model]
            np.testing.assert_array_equal(model_scores['atlas'], fmri_atlases)
            y, yerr = model_scores['score'], model_scores['error']
            return y, yerr

        _plot_bars(ax, models=models, get_model_score=get_model_score, ylim=ylim)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', which='both', length=0)  # hide ticks
        ax.tick_params(axis='x', pad=3)  # shift labels
        if ax_iter == 0:
            ax.set_xticklabels(fmri_atlases, fontdict=dict(fontsize=12))
        else:
            ax.set_xticklabels([])

    fig.subplots_adjust(wspace=0, hspace=.2)
    pyplot.savefig(Path(__file__).parent / f'bars-{benchmark}.png', dpi=600)


def ecog_best(models=all_models, benchmark='Fedorenko2016-encoding'):
    # ordering
    fmri_data = collect_best_scores(benchmark='Pereira2018-encoding', models=models)
    mean_scores = fmri_data.groupby('model')['score'].mean()
    models = mean_scores.sort_values().index.values
    # plot
    data = collect_best_scores(benchmark, models=models)
    ylim = 0.30
    fig, ax = pyplot.subplots(figsize=(3, 4))
    ax.set_title('ECoG')

    def get_model_score(model):
        model_score = data[data['model'] == model]
        y, yerr = model_score['score'], model_score['error']
        return y, yerr

    _plot_bars(ax, models=models, get_model_score=get_model_score, ylim=ylim,
               text_kwargs=dict(fontdict=dict(fontsize=9)))
    ax.set_xticks([])
    ax.set_xticklabels([])
    fig.tight_layout()
    pyplot.savefig(Path(__file__).parent / f'bars-{benchmark}.png', dpi=600)


def _plot_bars(ax, models, get_model_score, ylim, width=0.5, text_kwargs=None):
    text_kwargs = {**dict(fontdict=dict(fontsize=7), color='white'), **(text_kwargs or {})}
    step = (len(models) + 1) * width
    offset = len(models) / 2
    for model_iter, model in enumerate(models):
        y, yerr = get_model_score(model)
        x = np.arange(start=0, stop=len(y) * step, step=step)
        model_x = x - offset * width + model_iter * width
        ax.bar(model_x, height=y, yerr=yerr, width=width, edgecolor='none', color=model_colors[model])
        for xpos in model_x:
            ax.text(x=xpos + .8 * width / 2, y=.01, s=model,
                    rotation=90, rotation_mode='anchor', **text_kwargs)
    ax.set_ylabel("Predictivity (Pearson r)", fontdict=dict(fontsize=10))
    ax.set_ylim([0, ylim])
    ax.set_yticks(np.arange(0, ylim, .1))
    ax.set_yticklabels(["0" if label == 0 else f"{label:.1f}"[1:] if label % 0.2 == 0 else ""
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
