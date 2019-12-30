import fire
import logging
import numpy as np
import seaborn
import sys
from matplotlib import pyplot
from pathlib import Path

from neural_nlp.analyze.scores import models as all_models, collect_scores

seaborn.set(context='talk')


def best(models=all_models, benchmark='Pereira2018-encoding'):
    data = collect_best_scores(benchmark, models)
    width = 0.5
    step = (len(models) + 1) * width
    ylim = 0.35
    # ordering
    mean_scores = data.groupby('model')['score'].mean()
    models = mean_scores.sort_values().index.values
    atlases = ('DMN', 'MD', 'language', 'auditory', 'visual')
    assert set(data['atlas'].values) == set(atlases)
    experiments = ('384sentences', '243sentences')
    assert set(data['experiment'].values) == set(experiments)
    # plot
    fig, axes = pyplot.subplots(nrows=2)
    for ax_iter, (experiment, ax) in enumerate(zip(experiments, axes)):
        ax.set_title(experiment)
        experiment_scores = data[data['experiment'] == experiment]
        offset = len(models) / 2
        x = np.arange(start=0, stop=len(atlases) * step, step=step)
        for model_iter, model in enumerate(models):
            model_scores = experiment_scores[experiment_scores['model'] == model]
            y, yerr = model_scores['score'], model_scores['error']
            model_x = x - offset * width + model_iter * width
            ax.bar(model_x, height=y, yerr=yerr, width=width, edgecolor='none')
            for xpos in model_x:
                ax.text(x=xpos + .8 * width / 2, y=.01, s=model,
                        rotation=90, rotation_mode='anchor', fontdict=dict(fontsize=8), color='white')
        ax.set_ylabel("Predictivity (Pearson r)", fontdict=dict(fontsize=10))
        ax.set_ylim([0, ylim])
        ax.set_yticks(np.arange(0, ylim, .1))
        ax.set_yticklabels(["0" if label == 0 else f"{label:.1f}"[1:] if label % 0.2 == 0 else ""
                            for label in ax.get_yticks()], fontdict=dict(fontsize=14))
        ax.set_xticks(x)
        ax.tick_params(axis="x", pad=-5)
        if ax_iter == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(atlases, fontdict=dict(fontsize=14))

    fig.subplots_adjust(wspace=0, hspace=.2)
    pyplot.savefig(Path(__file__).parent / f'bars-{benchmark}.png', dpi=600)


def collect_best_scores(benchmark, models):
    scores = collect_scores(benchmark=benchmark, models=models)
    scores = scores.loc[scores.groupby(['model', 'experiment', 'atlas'])['score'].idxmax()]  # max layer
    return scores


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()
