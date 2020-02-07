import os

import fire
import itertools
import logging
import matplotlib
import numpy as np
import pandas as pd
import seaborn
import sys
from functools import reduce
from matplotlib import pyplot
from matplotlib.colors import to_rgba
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from neural_nlp import score, model_layers, benchmark_pool
from neural_nlp.analyze.sampled_architectures.neural_scores import score_all_models, \
    _score_model as score_architecture_model
from neural_nlp.benchmarks.neural import aggregate
from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
from neural_nlp.utils import ordered_set
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
    'xlm-roberta-base': 'magenta',
    'xlm-roberta-large': 'm',
}
models = tuple(model_colors.keys())

fmri_atlases = ('DMN', 'MD', 'language', 'auditory', 'visual')
overall_benchmarks = ('Pereira2018-encoding', 'Fedorenko2016-encoding', 'stories_froi_bold4s-encoding')


def compare(benchmark1='Pereira2018-encoding', benchmark2='Pereira2018-rdm', best_layer=True, normalize=True):
    all_models = models
    scores1 = collect_scores(benchmark=benchmark1, models=all_models)
    scores2 = collect_scores(benchmark=benchmark2, models=all_models)
    scores1, scores2 = average_adjacent(scores1).dropna(), average_adjacent(scores2).dropna()
    if best_layer:
        scores1, scores2 = choose_best_scores(scores1), choose_best_scores(scores2)
    if normalize:
        scores1, scores2 = ceiling_normalize(scores1), ceiling_normalize(scores2)
    scores1, scores2 = align_scores(scores1, scores2, identifier_set=['model'] if best_layer else ['model', 'layer'])
    colors = [model_colors[model.replace('-untrained', '')] for model in scores1['model'].values]
    colors = [to_rgba(named_color) for named_color in colors]
    savename = f"{benchmark1}__{benchmark2}"
    fig, ax = _plot_scores1_2(scores1, scores2, color=colors, alpha=None if best_layer else .2,
                              score_annotations=scores1['model'].values if best_layer else None,
                              xlabel=benchmark1, ylabel=benchmark2, loss_xaxis=benchmark1.startswith('wikitext'))
    savefig(fig, savename=savename)


def _plot_scores1_2(scores1, scores2, score_annotations=None, plot_correlation=True,
                    xlabel=None, ylabel=None, loss_xaxis=False, color=None, **kwargs):
    assert len(scores1) == len(scores2)
    x, xerr = scores1['score'].values, scores1['error'].values
    y, yerr = scores2['score'].values, scores2['error'].values
    fig, ax = pyplot.subplots()
    ax.scatter(x=x, y=y, c=color, s=2)
    ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, fmt='none', marker=None, ecolor=color, **kwargs)
    if score_annotations is not None:
        for annotation, _x, _y in zip(score_annotations, x, y):
            ax.text(_x, _y, annotation, fontdict=dict(fontsize=10), zorder=100)

    if loss_xaxis:
        ax.set_xlim(list(reversed(ax.get_xlim())))  # flip x

        @matplotlib.ticker.FuncFormatter
        def loss_formatter(loss, pos):
            return f"{loss}\n{np.exp(loss):.0f}"

        ax.xaxis.set_major_formatter(loss_formatter)

    for i, (name, correlate) in enumerate([('pearson', pearsonr), ('spearman', spearmanr)]):
        r, p = correlate(x, y)
        if i == 0 and plot_correlation:
            b, m = polyfit(x, y, 1)
            correlation_x = [min(x), max(x)]
            ax.plot(correlation_x, b + m * np.array(correlation_x))
        ax.text(0.9, 0.2 - i * 0.1, ha='center', va='center', transform=ax.transAxes,
                s=f"{name} " + ((f"r={(r * (-1 if loss_xaxis else 1)):.2f}" + significance_stars(p))
                                if p < 0.05 else f"n.s., p={p:.2f}"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def significance_stars(p, max_stars=5):
    return '*' * max([i for i in range(max_stars + 1) if p <= 0.5 / (10 ** i)])


def sampled_architectures(zoo_dir='/braintree/data2/active/users/msch/zoo.wmt17-lm',
                          benchmark='Pereira2018-encoding-min'):
    scores = score_all_models(zoo_dir, benchmark=benchmark, perplexity_threshold=20)
    model_dirs, scores = list(scores.keys()), list(scores.values())
    architectures = [model_dir.name[:5] for model_dir in model_dirs]
    zoo_name = Path(zoo_dir).name
    fig, ax = _bars(architectures, scores, ylabel=f"MT model scores on {benchmark}")
    ax.set_ylim([ax.get_ylim()[0], 0.3])
    savefig(fig, savename=f"{zoo_name}-{benchmark}")


def lstm_mt_vs_lm(benchmark='Pereira2018-encoding-min'):
    mt_score = score_architecture_model('/braintree/data2/active/users/msch/zoo.wmt17/'  # this is actually -mt
                                        'eb082aa158bb3ae2f9c5c3d4c5ff7bae2f93f901',
                                        benchmark=benchmark)
    lm_score = score(model='lm_1b', benchmark=benchmark)
    fig, ax = _bars(['MT: WMT\'17', 'LM: 1B'], [mt_score, lm_score], fig_kwargs=dict(figsize=(5, 5)))
    ax.set_ylim([ax.get_ylim()[0], 0.3])
    ax.set_title('LSTM trained on Machine Translation/Language Modeling')
    savefig(fig, 'lstm_mt_lm')


def collect_scores(benchmark, models):
    store_file = Path(__file__).parent / f'scores-{benchmark}.csv'
    if store_file.is_file():
        data = pd.read_csv(store_file)
        data = data[data['model'].isin(models)]
        return data
    if benchmark == 'overall':
        data = [collect_scores(benchmark=b, models=models) for b in
                ['Pereira2018-encoding', 'Fedorenko2016-encoding', 'stories_froi_bold4s-encoding']]
        data = reduce(lambda left, right: pd.concat([left, right]), data)
        data = average_adjacent(data)
        data = data.groupby(['model', 'layer']).mean().reset_index()  # mean across benchmarks per model-layer
        data['benchmark'] = 'overall'
    else:
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
                center, error = get_score_center_err(current_score)
                data.append(
                    {**adjunct_values, **{'benchmark': benchmark, 'model': model, 'score': center, 'error': error}})
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
    savefig(fig, 'brain_network_correlations')


def align_both(data1, data2, on):
    data1 = data1[data1[on].isin(data2[on])]
    data2 = data2[data2[on].isin(data1[on])]
    data1 = data1.set_index(on).reindex(index=data2[on]).reset_index()
    return data1, data2


def untrained_vs_trained(benchmark='Pereira2018-encoding', layer_mode='best'):
    """
    :param layer_mode: 'best' to select the best layer per model,
      'group' to keep all layers and color them based on their model,
      'pos' to keep all layers and color them based on their relative position.
    """
    all_models = [[model, f"{model}-untrained"] for model in models]
    all_models = [model for model_tuple in all_models for model in model_tuple]
    scores = collect_scores(benchmark=benchmark, models=all_models)
    scores = average_adjacent(scores)  # average experiments & atlases
    scores = scores.dropna()  # embedding layers in xlnets and t5s have nan scores
    if layer_mode == 'best':
        scores = choose_best_scores(scores)
    elif layer_mode == 'pos':
        scores['layer_position'] = [model_layers[model].index(layer) / len(model_layers[model])
                                    for model, layer in zip(scores['model'].values, scores['layer'].values)]
    scores = ceiling_normalize(scores)
    # separate into trained / untrained
    untrained_rows = np.array([model.endswith('-untrained') for model in scores['model']])
    scores_trained, scores_untrained = scores[~untrained_rows], scores[untrained_rows]
    # align
    scores_untrained['model_identifier'] = [model.replace('-untrained', '')
                                            for model in scores_untrained['model'].values]
    if layer_mode == 'best':  # layer is already argmax'ed over, might not be same across untrained/trained
        identifiers_trained = scores_trained['model'].values
        identifiers_untrained = scores_untrained['model_identifier'].values
    else:
        identifiers_trained = list(zip(scores_trained['model'].values, scores_trained['layer'].values))
        identifiers_untrained = list(zip(scores_untrained['model_identifier'].values, scores_untrained['layer'].values))
    overlap = set(identifiers_trained).intersection(set(identifiers_untrained))
    scores_trained = scores_trained[[identifier in overlap for identifier in identifiers_trained]]
    scores_untrained = scores_untrained[[identifier in overlap for identifier in identifiers_untrained]]
    scores_trained = scores_trained.sort_values(['model', 'layer'])
    scores_untrained = scores_untrained.sort_values(['model_identifier', 'layer'])
    if layer_mode != 'best':
        assert (scores_trained['layer'].values == scores_untrained['layer'].values).all()
    # plot
    if layer_mode in ('best', 'group'):
        colors = [model_colors[model] for model in scores_trained['model']]
        colors = [to_rgba(named_color) for named_color in colors]
    else:
        cmap = matplotlib.cm.get_cmap('binary')
        colors = cmap(scores_trained['layer_position'].values)
    fig, ax = _plot_scores1_2(scores_untrained, scores_trained, alpha=None if layer_mode == 'best' else 0.4,
                              color=colors, xlabel="untrained", ylabel="trained")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='dashed', color='darkgray')
    savefig(fig, savename=f"untrained_trained-{benchmark}")


def choose_best_scores(scores):
    adjunct_columns = list(set(scores.columns) - {'score', 'error', 'layer'})
    scores = scores.loc[scores.groupby(adjunct_columns)['score'].idxmax()]  # max layer
    return scores


def num_features_vs_score(benchmark='Pereira2018-encoding', per_layer=True, include_untrained=True):
    if include_untrained:
        all_models = [(model, f"{model}-untrained") for model in models]
        all_models = [model for model_tuple in all_models for model in model_tuple]
    else:
        all_models = models
    scores = collect_scores(benchmark=benchmark, models=all_models)
    scores = average_adjacent(scores)
    scores = scores.dropna()
    if not per_layer:
        scores = choose_best_scores(scores)
    # count number of features
    num_features = []
    for model in tqdm(ordered_set(scores['model'].values), desc='models'):
        # mock-run stimuli that are already stored
        mock_extractor = ActivationsExtractorHelper(get_activations=None, reset=None)
        features = mock_extractor._from_sentences_stored(
            layers=model_layers[model.replace('-untrained', '')], sentences=None,
            identifier=model.replace('-untrained', ''), stimuli_identifier='Pereira2018-243sentences.astronaut')
        if per_layer:
            for layer in scores['layer'].values[scores['model'] == model]:
                num_features.append({'model': model, 'layer': layer,
                                     'score': len(features.sel(layer=layer)['neuroid'])})
        else:
            num_features.append({'model': model, 'score': len(features['neuroid'])})
    num_features = pd.DataFrame(num_features)
    num_features['error'] = np.nan
    if per_layer:
        assert (scores['layer'].values == num_features['layer'].values).all()
    # plot
    colors = [model_colors[model.replace('-untrained', '')] for model in scores['model'].values]
    fig, ax = _plot_scores1_2(num_features, scores, color=colors, xlabel="number of features", ylabel=benchmark)
    savefig(fig, savename=f"num_features-{benchmark}" + ("-layerwise" if per_layer else ""))


def average_adjacent(data, keep_columns=('benchmark', 'model', 'layer'), skipna=False):
    data = data.groupby(list(keep_columns)).agg(lambda g: g.mean(skipna=skipna))  # mean across non-keep columns
    return data.reset_index()


def get_score_center_err(s, combine_layers=True):
    s = aggregate(s, combine_layers=combine_layers)
    if hasattr(s, 'aggregation'):
        return s.sel(aggregation='center').values.tolist(), s.sel(aggregation='error').values.tolist()
    if hasattr(s, 'measure'):
        # s = s.sel(measure='test_perplexity') if len(s['measure'].values.shape) > 0 else s
        s = s.sel(measure='test_loss') if len(s['measure'].values.shape) > 0 else s
        s = s.values.tolist()
        return s, np.nan
    if isinstance(s, (int, float)):
        return s, np.nan
    raise ValueError(f"Unknown score structure: {s}")


def ceiling_normalize(scores):
    scores['score_unceiled'] = scores['score']
    benchmark_ceilings = {}
    for benchmark in set(scores['benchmark'].values):
        if benchmark == 'overall':
            ceilings = [benchmark_pool[part]().ceiling.sel(aggregation='center') for part in overall_benchmarks]
            benchmark_ceilings[benchmark] = np.mean(ceilings)
        else:
            benchmark_ceilings[benchmark] = benchmark_pool[benchmark]().ceiling.sel(aggregation='center') \
                .values.tolist()
    scores['ceiling'] = [benchmark_ceilings[benchmark] for benchmark in scores['benchmark'].values]
    scores['score'] = scores['score'] / scores['ceiling']
    return scores


def savefig(fig, savename):
    fig.tight_layout()
    savepath = Path(__file__).parent / f"{savename}.png"
    logger.info(f"Saving to {savepath}")
    fig.savefig(savepath, dpi=600)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    seaborn.set(context='talk')
    fire.Fire()
