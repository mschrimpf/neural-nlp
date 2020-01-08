import os
from decimal import Decimal

import fire
import itertools
import logging
import numpy as np
import pandas as pd
import seaborn
import sys
from matplotlib import pyplot
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm

from neural_nlp import score, model_layers
from neural_nlp.analyze.sampled_architectures.neural_scores import score_all_models, \
    _score_model as score_architecture_model
from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
from result_caching import NotCachedError

logger = logging.getLogger(__name__)

model_colors = {
    # controls / embeddings
    'sentence-length': 'lightgray',
    'word2vec': 'silver',
    'skip-thoughts': 'darkgray',
    'glove': 'gray',
    # semantic?
    'topicETM': 'bisque',
    # LSTMs
    'lm_1b': 'slategrey',
    # naive transformer
    'transformer': 'rosybrown',
    # BERT
    'bert-base-uncased': 'tomato',
    'bert-base-multilingual-cased': 'r',
    'bert-large-uncased': 'orangered',
    'bert-large-uncased-whole-word-masking': 'red',
    'distilbert-base-uncased': 'salmon',
    # RoBERTa
    'distilroberta-base': 'firebrick',
    'roberta-base': 'brown',
    'roberta-large': 'maroon',
    # GPT
    'distilgpt2': 'lightblue',
    'openaigpt': 'cadetblue',
    'gpt2': 'steelblue',
    'gpt2-medium': 'c',
    'gpt2-large': 'teal',
    'gpt2-xl': 'darkslategray',
    # Transformer-XL
    'transfo-xl-wt103': 'peru',
    # XLNet
    'xlnet-base-cased': 'yellow',
    'xlnet-large-cased': 'gold',
    # XLM
    'xlm-mlm-en-2048': 'orange',
    'xlm-mlm-enfr-1024': 'darkorange',
    'xlm-mlm-xnli15-1024': 'goldenrod',
    'xlm-clm-enfr-1024': 'chocolate',
    'xlm-mlm-100-1280': 'darkgoldenrod',
    # CTRL
    'ctrl': 'blue',
    # AlBERT
    'albert-base-v1': 'limegreen',
    'albert-base-v2': 'limegreen',
    'albert-large-v1': 'forestgreen',
    'albert-large-v2': 'forestgreen',
    'albert-xlarge-v1': 'green',
    'albert-xlarge-v2': 'green',
    'albert-xxlarge-v1': 'darkgreen',
    'albert-xxlarge-v2': 'darkgreen',
    # T5
    't5-small': 'mediumpurple',
    't5-base': 'blueviolet',
    't5-large': 'mediumorchid',
    't5-3b': 'darkviolet',
    't5-11b': 'rebeccapurple',
    # XLM-RoBERTa
    'xlm-roberta-base': 'magenta', 'xlm-roberta-large': 'm',
}
models = tuple(model_colors.keys())

fmri_atlases = ('DMN', 'MD', 'language', 'auditory', 'visual')


def compare(benchmark1='Pereira2018-encoding', benchmark2='Pereira2018-rdm', flip_x=False):
    scores1, scores2, run_models = [], [], []
    os.environ['RESULTCACHING_CACHEDONLY'] = '1'
    for model in models:
        try:
            score1 = score(benchmark=benchmark1, model=model)
            score2 = score(benchmark=benchmark2, model=model)
            scores1.append(score1)
            scores2.append(score2)
            run_models.append(model)
        except NotCachedError:
            continue
    savename = f"{benchmark1}__{benchmark2}"
    fig, ax = _plot_scores1_2(scores1, scores2, score_annotations=run_models,
                              xlabel=benchmark1, ylabel=benchmark2, flip_x=flip_x)
    _savefig(fig, savename=savename)


def _plot_scores1_2(scores1, scores2, score_annotations=None, xlabel=None, ylabel=None, flip_x=False, **kwargs):
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
        if isinstance(s, (int, float)):
            return s, 0
        raise ValueError(f"Unknown score structure: {s}")

    x, xerr = [get_center_err(s)[0] for s in scores1], [get_center_err(s)[1] for s in scores1]
    y, yerr = [get_center_err(s)[0] for s in scores2], [get_center_err(s)[1] for s in scores2]
    fig, ax = pyplot.subplots()
    ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, fmt='.', **kwargs)
    if score_annotations:
        for annotation, _x, _y in zip(score_annotations, x, y):
            ax.text(_x, _y, annotation, fontdict=dict(fontsize=10))

    if flip_x:
        ax.set_xlim(list(reversed(ax.get_xlim())))

    correlation, p = pearsonr(x, y)
    b, m = polyfit(x, y, 1)
    ax.plot(ax.get_xlim(), b + m * np.array(ax.get_xlim()))
    ax.text(0.9, 0.1, ha='center', va='center', transform=ax.transAxes,
            s=(f"r={(correlation * (-1 if flip_x else 1)):.2f}" +
               significance_stars(p))
            if p < 0.05 else f"n.s., p={p:.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def significance_stars(p):
    return '*' * max([i for i in range(5) if p <= 0.5 / (10 ** i)])


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
        os.environ['RESULTCACHING_CACHEDONLY'] = '1'
        try:
            model_scores = score(benchmark=benchmark, model=model)
        except NotCachedError:
            continue
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
    ax.text(0.8, 0.1, "r: " + ((f"{r:.2f}" + significance_stars(p)) if p < .05 else "n.s."),
            ha='center', va='center', transform=ax.transAxes)
    ticks = np.arange(0, 0.4, 0.05)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ticklabels = ["0" if tick == 0 else f"{tick:.1f}"[1:] if Decimal(f"{tick:.2f}") % Decimal(".1") == 0 else ""
                  for tick in ticks]
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
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


def untrained_vs_trained(benchmark='Pereira2018-encoding'):
    os.environ['RESULTCACHING_CACHEDONLY'] = '1'
    trained_scores, untrained_scores, run_models = [], [], []
    for model in models:
        try:
            trained_score = score(benchmark=benchmark, model=model)
            untrained_score = score(benchmark=benchmark, model=f"{model}-untrained")
            trained_scores.append(trained_score)
            untrained_scores.append(untrained_score)
            run_models.append(model)
        except NotCachedError:
            continue
    fig, ax = _plot_scores1_2(untrained_scores, trained_scores, score_annotations=run_models,
                              xlabel="untrained", ylabel="trained")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='dashed', color='darkgray')
    _savefig(fig, savename=f"untrained_trained-{benchmark}")


def num_features_vs_score(benchmark='Pereira2018-encoding', models=models, per_layer=False):
    num_features, scores, colors = [], [], None
    for model in models:
        model_score = score(benchmark=benchmark, model=model)
        # mock-run stimuli that are already stored
        mock_extractor = ActivationsExtractorHelper(get_activations=None, reset=None)
        features = mock_extractor._from_sentences_stored(
            layers=model_layers[model], sentences=None,
            identifier=model, stimuli_identifier='Pereira2018-243sentences.astronaut')
        if per_layer:
            colors = []
            for layer in model_layers[model]:
                num_features.append(len(features.sel(layer=layer)['neuroid']))
                scores.append(model_score.sel(layer=layer, drop=True))
                colors.append(model_colors[model])
        else:
            num_features.append(len(features['neuroid']))
            scores.append(model_score)
    fig, ax = _plot_scores1_2(num_features, scores, color=colors, xlabel="number of features", ylabel=benchmark)
    _savefig(fig, savename=f"num_features-{benchmark}")


def _savefig(fig, savename):
    fig.tight_layout()
    savepath = Path(__file__).parent / f"{savename}.png"
    logger.info(f"Saving to {savepath}")
    fig.savefig(savepath, dpi=600)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
