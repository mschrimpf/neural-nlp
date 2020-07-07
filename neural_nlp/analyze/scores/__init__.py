import fire
import itertools
import logging
import matplotlib
import numpy as np
import os
import pandas as pd
import seaborn
import sys
from functools import reduce
from matplotlib import pyplot
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator
from numpy.polynomial.polynomial import polyfit
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from neural_nlp import score, model_layers, benchmark_pool
from neural_nlp.analyze import savefig, score_formatter
from neural_nlp.benchmarks.neural import aggregate
from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
from neural_nlp.utils import ordered_set
from result_caching import NotCachedError, is_iterable

logger = logging.getLogger(__name__)

model_colors = {
    # embeddings
    'glove': 'gray',
    'ETM': 'bisque',
    'word2vec': 'silver',
    # RNNs
    'lm_1b': 'slategrey',
    'skip-thoughts': 'darkgray',
    # BERT
    'distilbert-base-uncased': 'salmon',
    'bert-base-uncased': 'tomato',
    'bert-base-multilingual-cased': 'r',
    'bert-large-uncased': 'orangered',
    'bert-large-uncased-whole-word-masking': 'red',
    # RoBERTa
    'distilroberta-base': 'firebrick',
    'roberta-base': 'brown',
    'roberta-large': 'maroon',
    # XLM
    'xlm-mlm-enfr-1024': 'darkorange',
    'xlm-clm-enfr-1024': 'chocolate',
    'xlm-mlm-xnli15-1024': 'goldenrod',
    'xlm-mlm-100-1280': 'darkgoldenrod',
    'xlm-mlm-en-2048': 'orange',
    # XLM-RoBERTa
    'xlm-roberta-base': '#bc6229',
    'xlm-roberta-large': '#974e20',
    # Transformer-XL
    'transfo-xl-wt103': 'peru',
    # XLNet
    'xlnet-base-cased': 'gold',
    'xlnet-large-cased': '#ffbf00',
    # CTRL
    'ctrl': '#009EDB',
    # T5
    't5-small': 'mediumorchid',
    't5-base': 'mediumpurple',
    't5-large': 'blueviolet',
    't5-3b': 'darkviolet',
    't5-11b': 'rebeccapurple',
    # AlBERT
    'albert-base-v1': 'limegreen',
    'albert-base-v2': 'limegreen',
    'albert-large-v1': 'forestgreen',
    'albert-large-v2': 'forestgreen',
    'albert-xlarge-v1': 'green',
    'albert-xlarge-v2': 'green',
    'albert-xxlarge-v1': 'darkgreen',
    'albert-xxlarge-v2': 'darkgreen',
    # GPT
    'openaigpt': 'lightblue',
    'distilgpt2': 'c',
    'gpt2': 'cadetblue',
    'gpt2-medium': 'steelblue',
    'gpt2-large': 'teal',
    'gpt2-xl': 'darkslategray',
}
models = tuple(model_colors.keys())

fmri_atlases = ('DMN', 'MD', 'language', 'auditory', 'visual')
overall_neural_benchmarks = ('Pereira2018', 'Fedorenko2016v3', 'Blank2014fROI')
overall_benchmarks = overall_neural_benchmarks + ('Futrell2018',)
performance_benchmarks = ['wikitext', 'glue']


class LabelReplace(dict):
    def __missing__(self, key):
        return key


class BenchmarkLabelReplace(LabelReplace):
    def __init__(self):
        super(BenchmarkLabelReplace, self).__init__(**{
            'overall_neural': 'Language Brain-Score (neural)',
            'overall': 'Language Brain-Score',
            'Blank2014fROI': 'Blank2014',
            'Fedorenko2016v3': 'Fedorenko2016',
            'wikitext-2': 'wikitext-2 perplexity',
        })

    def __getitem__(self, item):
        if item.endswith('-encoding'):
            return super(BenchmarkLabelReplace, self).__getitem__(item[:-len('-encoding')])
        return super(BenchmarkLabelReplace, self).__getitem__(item)


benchmark_label_replace = BenchmarkLabelReplace()
model_label_replace = LabelReplace({'word2vec': 'w2v', 'transformer': 'trf.', 'lm_1b': 'lstm lm1b'})


def compare(benchmark1='wikitext-2', benchmark2='Blank2014fROI-encoding',
            best_layer=True, normalize=True, reference_best=False, identity_line=False, annotate=False,
            plot_ceiling=False, xlim=None, ylim=None, ax=None, **kwargs):
    ax_given = ax is not None
    all_models = models
    scores1 = collect_scores(benchmark=benchmark1, models=all_models, normalize=normalize)
    scores2 = collect_scores(benchmark=benchmark2, models=all_models, normalize=normalize)
    scores1, scores2 = average_adjacent(scores1).dropna(), average_adjacent(scores2).dropna()
    if best_layer:
        choose_best = choose_best_scores if not reference_best else reference_best_scores
        scores1, scores2 = choose_best(scores1), choose_best(scores2)
    scores1, scores2 = align_scores(scores1, scores2, identifier_set=['model'] if best_layer else ['model', 'layer'])
    colors = [model_colors[model.replace('-untrained', '')] for model in scores1['model'].values]
    colors = [to_rgba(named_color) for named_color in colors]
    if not best_layer or not annotate:
        score_annotations = None
    elif annotate is True:
        score_annotations = scores1['model'].values
    else:
        score_annotations = [model if model in annotate else None for model in scores1['model'].values]
    fig, ax = _plot_scores1_2(scores1, scores2, color=colors, alpha=None if best_layer else .2,
                              score_annotations=score_annotations,
                              xlabel=benchmark1, ylabel=benchmark2, loss_xaxis=benchmark1.startswith('wikitext'),
                              ax=ax, **kwargs)
    xlim_given, ylim_given = xlim is not None, ylim is not None
    xlim, ylim = ax.get_xlim() if not xlim_given else xlim, ax.get_ylim() if not ylim_given else ylim
    normalize_x = normalize and not any(benchmark1.startswith(perf_prefix) for perf_prefix in performance_benchmarks)
    normalize_y = normalize and not any(benchmark2.startswith(perf_prefix) for perf_prefix in performance_benchmarks)
    if normalize_x and not xlim_given:
        xlim = [0, 1.1]
    if normalize_y and not ylim_given:
        ylim = [0, 1.1]
    if normalize_x and plot_ceiling:
        ceiling_err = get_ceiling(benchmark1)
        shaded_errorbar(y=ylim, x=np.array([1, 1]), error=ceiling_err, ax=ax, vertical=True,
                        alpha=0, shaded_kwargs=dict(color='gray', alpha=.5))
    if normalize_y and plot_ceiling:
        ceiling_err = get_ceiling(benchmark2)
        shaded_errorbar(x=xlim, y=np.array([1, 1]), error=ceiling_err, ax=ax,
                        alpha=0, shaded_kwargs=dict(color='gray', alpha=.5))
    if identity_line:
        lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        if not xlim_given:
            xlim = lim
        if not ylim_given:
            ylim = lim
        ax.plot(lim, lim, linestyle='dashed', color='gray')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if not ax_given:
        savefig(fig, savename=Path(__file__).parent /
                              (f"{benchmark1}__{benchmark2}" + ('-best' if best_layer else '-layers')))


def _plot_scores1_2(scores1, scores2, score_annotations=None, plot_correlation=False, plot_significance_stars=True,
                    xlabel=None, ylabel=None, loss_xaxis=False, color=None, tick_locator_base=0.2,
                    scatter_size=20, errorbar_error_alpha=0.2,
                    prettify_xticks=True, correlation_pos=(0.05, 0.8), ax=None, **kwargs):
    assert len(scores1) == len(scores2)
    x, xerr = scores1['score'].values, scores1['error'].values
    y, yerr = scores2['score'].values, scores2['error'].values
    fig, ax = pyplot.subplots() if ax is None else (None, ax)
    markers, caps, bars = ax.errorbar(x=x, xerr=xerr, y=y, yerr=yerr, fmt='none', marker=None, ecolor=color, **kwargs)
    [bar.set_alpha(errorbar_error_alpha) for bar in bars]
    [cap.set_alpha(errorbar_error_alpha) for cap in caps]
    ax.scatter(x=x, y=y, c=color, s=scatter_size)
    if score_annotations is not None:
        for annotation, _x, _y in zip(score_annotations, x, y):
            if not annotation:
                continue
            ax.annotate(annotation, xy=(_x, _y), xytext=(_x + .05, _y + .05), size=10, zorder=1,
                        arrowprops=dict(lw=1, arrowstyle="-", color='black'))

    if loss_xaxis:
        ax.set_xlim(list(reversed(ax.get_xlim())))  # flip x

        @matplotlib.ticker.FuncFormatter
        def loss_formatter(loss, pos):
            return f"{np.exp(loss):.0f}"

        ax.set_xticks(np.log([100, 200, 400, 800, 1600, 3200]))
        ax.xaxis.set_major_formatter(loss_formatter)
    elif prettify_xticks:
        ax.xaxis.set_major_locator(MultipleLocator(base=tick_locator_base))
        ax.xaxis.set_major_formatter(score_formatter)
    ax.yaxis.set_major_locator(MultipleLocator(base=tick_locator_base))
    ax.yaxis.set_major_formatter(score_formatter)

    r, p = pearsonr(x if not loss_xaxis else np.exp(x), y)
    if plot_correlation:
        b, m = polyfit(x, y, 1)
        correlation_x = [min(x), max(x)]
        ax.plot(correlation_x, b + m * np.array(correlation_x), color='black', linestyle='solid')
    ax.text(*correlation_pos, ha='left', va='center', transform=ax.transAxes,
            s=("$r=" + f"{(r * (-1 if loss_xaxis else 1)):.2f}$"[1:] + ('' if p < 0.05 else f" (n.s.)")
               + (significance_stars(p) if plot_significance_stars and p < 0.05 else '')))
    if not plot_significance_stars:
        for label, func in zip(['pearson', 'spearman'], [pearsonr, spearmanr]):
            r, p = func(x if not loss_xaxis else np.exp(x), y)
            logger.info(f"{label} r={r}, p={p}")

    ax.set_xlabel(benchmark_label_replace[xlabel])
    ax.set_ylabel(benchmark_label_replace[ylabel])
    return fig, ax


def significance_stars(p, max_stars=5):
    return '*' * max([i for i in range(max_stars + 1) if p <= 0.5 / (10 ** i)])


def collect_scores(benchmark, models, normalize=True, score_hook=None):
    store_file = Path(__file__).parent / f"scores-{benchmark}-{'normalized' if normalize else 'raw'}" \
                                         f"{'-hook' if score_hook else ''}.csv"
    stored = False
    if store_file.is_file():
        data = pd.read_csv(store_file)
        data = data[data['model'].isin(models)]
        stored = True
    if not stored and benchmark.startswith('overall'):
        metric = benchmark.split('-')[-1]
        data = [collect_scores(benchmark=f"{part}-{metric}", models=models, normalize=normalize) for part in
                (overall_neural_benchmarks if benchmark.startswith('overall_neural') else overall_benchmarks)]
        data = reduce(lambda left, right: pd.concat([left, right]), data)
        data = average_adjacent(data)
        data = choose_best_scores(data).dropna()  # need to choose best layer per benchmark here before averaging
        data['score'][data['score'] >= 1] = 1  # more than 100% makes no sense and might skew averaging
        data = data.groupby(['model']).mean().reset_index()  # mean across benchmarks per model-layer
        data['layer'] = 'combined'
        data['benchmark'] = benchmark
    elif not stored:
        data, missing_models = [], []
        previous_resultcaching_cachedonly = os.getenv('RESULTCACHING_CACHEDONLY', '0')
        os.environ['RESULTCACHING_CACHEDONLY'] = '1'
        for model in tqdm(models, desc='model scores'):
            try:
                model_scores = score(benchmark=benchmark, model=model)
                if score_hook:
                    model_scores = score_hook(model_scores)
            except NotCachedError:
                missing_models.append(model)
                continue
            if not normalize:
                model_scores = model_scores.raw
            adjunct_columns = list(set(model_scores.dims) - {'aggregation'})
            for adjunct_values in itertools.product(*[model_scores[column].values for column in adjunct_columns]):
                adjunct_values = dict(zip(adjunct_columns, adjunct_values))
                current_score = model_scores.sel(**adjunct_values)
                center, error = get_score_center_err(current_score)
                data.append({**adjunct_values,
                             **{'benchmark': benchmark, 'model': model, 'score': center, 'error': error}})
        if missing_models:
            logger.warning(
                f"No score cached for {len(missing_models)} models {missing_models} on benchmark {benchmark}")
        data = pd.DataFrame(data)
        if any(benchmark.startswith(performance_benchmark) for performance_benchmark in ['wikitext', 'glue']):
            data['layer'] = -1
            data['error'] = 0  # nans will otherwise be dropped later on
        if benchmark.startswith('wikitext'):
            data = data[data['measure'] == 'test_loss']
        if benchmark == 'glue-mnli':  # only consider mnli (not mnli-mm)
            data = data[data['eval_task'] == 'mnli']
        os.environ['RESULTCACHING_CACHEDONLY'] = previous_resultcaching_cachedonly
    non_overlap = list(set(data['model']).symmetric_difference(set(models)))
    if len(non_overlap) > 0:
        logger.warning(f"Non-overlapping identifiers in {benchmark}: {sorted(non_overlap)}")
    if not stored:
        data.to_csv(store_file, index=False)
    return data


def compare_glue(benchmark2='Pereira2018-encoding'):
    from neural_nlp.benchmarks.glue import benchmark_pool as glue_benchmark_pool
    fig, axes = pyplot.subplots(figsize=(18, 8), nrows=2, ncols=4, sharey=True)
    for i, (ax, glue_benchmark) in enumerate(zip(axes.flatten(), glue_benchmark_pool)):
        compare(benchmark1=glue_benchmark, benchmark2=benchmark2, ax=ax, identity_line=False,
                correlation_pos=(0.5, 0.1), prettify_xticks=False)
        if i % 4 != 0:
            ax.set_ylabel(None)
    savefig(fig, Path(__file__).parent / f"glue-{benchmark2}")


def Pereira2018_experiment_correlations(best_layer=False, **kwargs):
    experiment2_scores, experiment3_scores = collect_Pereira_experiment_scores(best_layer)
    # plot
    colors = [model_colors[model.replace('-untrained', '')] for model in experiment2_scores['model'].values]
    colors = [to_rgba(named_color) for named_color in colors]
    fig, ax = pyplot.subplots(figsize=(6, 6))
    _plot_scores1_2(experiment2_scores, experiment3_scores, color=colors, alpha=None if best_layer else .2,
                    score_annotations=None, xlabel='Pereira2018 (Exp. 2)', ylabel='Pereira2018 (Exp. 3)', ax=ax,
                    **kwargs)
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 1.1])
    scores = np.concatenate((experiment2_scores['score'].values, experiment3_scores['score'].values))
    ax.plot([min(scores), max(scores)], [min(scores), max(scores)], linestyle='dashed', color='gray')
    savefig(fig, savename=Path(__file__).parent / ('fmri-correlations' + ('-best' if best_layer else '-layers')))


def collect_Pereira_experiment_scores(best_layer=False):
    from brainscore.metrics import Score
    from scipy.stats import median_absolute_deviation
    from neural_nlp.benchmarks.neural import consistency

    def get_experiment_score(score):
        raw_score = score.raw.raw.raw
        raw_score = raw_score.mean('split')
        # redo what is done in `aggregate_neuroid_scores`, but keeping the experiment dimension
        subject_scores = raw_score.groupby('subject').median('neuroid')
        center = subject_scores.median('subject')
        subject_values = np.nan_to_num(subject_scores.values, nan=0)
        error = median_absolute_deviation(subject_values, axis=subject_scores.dims.index('subject'))
        error = center.__class__(error, coords=center.coords, dims=center.dims)
        center = consistency(center, score.ceiling.sel(aggregation='center').values)
        error = consistency(error, score.ceiling.sel(aggregation='center').values)
        center, error = center.expand_dims('aggregation'), error.expand_dims('aggregation')
        center['aggregation'], error['aggregation'] = ['center'], ['error']
        score = Score.merge(center, error)
        return score

    # use best layer based on base average
    base_scores = collect_scores(benchmark='Pereira2018-encoding', models=models)
    base_scores = choose_best_scores(base_scores)
    layers = set((row.model, row.layer) for row in base_scores.itertuples())
    # now collect per-experiment scores and sub-select layer based on base scores
    scores = collect_scores(benchmark='Pereira2018-encoding', models=models, score_hook=get_experiment_score)
    scores = average_adjacent(scores, keep_columns=['benchmark', 'model', 'layer', 'experiment'])
    scores = scores.dropna()
    scores = scores[[(row.model, row.layer) in layers for row in scores.itertuples()]]
    # separate into experiments & align
    experiment2_scores = scores[scores['experiment'] == '384sentences']
    experiment3_scores = scores[scores['experiment'] == '243sentences']
    experiment2_scores, experiment3_scores = align_scores(
        experiment2_scores, experiment3_scores, identifier_set=('model',) if best_layer else ('model', 'layer'))
    return experiment2_scores, experiment3_scores


def align_scores(scores1, scores2, identifier_set=('model', 'layer')):
    identifiers1 = list(zip(*[scores1[identifier_key].values for identifier_key in identifier_set]))
    identifiers2 = list(zip(*[scores2[identifier_key].values for identifier_key in identifier_set]))
    overlap = list(set(identifiers1).intersection(set(identifiers2)))
    non_overlap = list(set(identifiers1).symmetric_difference(set(identifiers2)))
    if len(non_overlap) > 0:
        logger.warning(f"Non-overlapping identifiers: {sorted(non_overlap)}")
    scores1 = scores1.iloc[[identifiers1.index(identifier) for identifier in overlap]]
    scores2 = scores2.iloc[[identifiers2.index(identifier) for identifier in overlap]]
    return scores1, scores2


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
    savefig(fig, Path(__file__).parent / 'brain_network_correlations')


def align_both(data1, data2, on):
    data1 = data1[data1[on].isin(data2[on])]
    data2 = data2[data2[on].isin(data1[on])]
    data1 = data1.set_index(on).reindex(index=data2[on]).reset_index()
    return data1, data2


def untrained_vs_trained(benchmark='Pereira2018-encoding', layer_mode='best', model_selection=None,
                         analyze_only=False, **kwargs):
    """
    :param layer_mode: 'best' to select the best layer per model,
      'group' to keep all layers and color them based on their model,
      'pos' to keep all layers and color them based on their relative position.
    """
    all_models = model_selection or models
    all_models = [[model, f"{model}-untrained"] for model in all_models]
    all_models = [model for model_tuple in all_models for model in model_tuple]
    scores = collect_scores(benchmark=benchmark, models=all_models)
    scores = average_adjacent(scores)  # average experiments & atlases
    scores = scores.dropna()  # embedding layers in xlnets and t5s have nan scores
    if layer_mode == 'best':
        scores = choose_best_scores(scores)
    elif layer_mode == 'pos':
        scores['layer_position'] = [model_layers[model].index(layer) / len(model_layers[model])
                                    for model, layer in zip(scores['model'].values, scores['layer'].values)]
    # separate into trained / untrained
    untrained_rows = np.array([model.endswith('-untrained') for model in scores['model']])
    scores_trained, scores_untrained = scores[~untrained_rows], scores[untrained_rows]
    # align
    scores_untrained['model'] = [model.replace('-untrained', '') for model in scores_untrained['model'].values]
    scores_trained, scores_untrained = align_scores(
        scores_trained, scores_untrained, identifier_set=('model',) if layer_mode == 'best' else ('model', 'layer'))
    if layer_mode != 'best':
        assert (scores_trained['layer'].values == scores_untrained['layer'].values).all()
    # analyze
    average_trained, average_untrained = np.mean(scores_trained['score']), np.mean(scores_untrained['score'])
    _, p_diff = pearsonr(scores_trained['score'], scores_untrained['score'])
    logger.info(f"Trained/untrained on {benchmark}: "
                f"score trained={average_trained:.2f}, untrained={average_untrained:.2f} | "
                f"diff {average_trained-average_untrained:.2f} ({average_trained/average_untrained * 100:.0f}%, "
                f"p={p_diff}")
    if analyze_only:
        return
    # plot
    if layer_mode in ('best', 'group'):
        colors = [model_colors[model] for model in scores_trained['model']]
        colors = [to_rgba(named_color) for named_color in colors]
    else:
        cmap = matplotlib.cm.get_cmap('binary')
        colors = cmap(scores_trained['layer_position'].values)
    fig, ax = pyplot.subplots(figsize=(6, 6))
    _plot_scores1_2(scores_untrained, scores_trained, alpha=None if layer_mode == 'best' else 0.4,
                    color=colors, xlabel="architecture (no training)", ylabel="architecture + training",
                    plot_significance_stars=False, ax=ax, **kwargs)
    lims = [-.05, 1.1] if benchmark.startswith('Fedorenko') else [-.05, 1.2] if benchmark.startswith('Pereira') \
        else [8, 4] if benchmark.startswith('wikitext-2') else [0, 1.335] if benchmark.startswith('Futrell') \
        else [-.05, 1.]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='dashed', color='darkgray')
    ax.set_title(benchmark_label_replace[benchmark])
    savefig(fig, savename=Path(__file__).parent / f"untrained_trained-{benchmark}")


def reference_best_scores(scores, reference_benchmark='Pereira2018-encoding'):
    reference_scores = collect_scores(benchmark=reference_benchmark, models=set(scores['model'].values))
    reference_scores = average_adjacent(reference_scores).dropna()
    best_reference_scores = choose_best_scores(reference_scores)
    selection_columns = ['model', 'layer']
    best_selection = set([tuple(identifier) for identifier in best_reference_scores[selection_columns].values])
    index = [tuple(identifier) in best_selection for identifier in scores[selection_columns].values]
    scores = scores[index]
    return scores


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
    store_file = Path(__file__).parent / "num_features.csv"
    if store_file.is_file():
        num_features = pd.read_csv(store_file)
    else:
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
        num_features.to_csv(store_file, index=False)
    if per_layer:
        assert (scores['layer'].values == num_features['layer'].values).all()
    # plot
    colors = [model_colors[model.replace('-untrained', '')] for model in scores['model'].values]
    fig, ax = _plot_scores1_2(num_features, scores, color=colors, xlabel="number of features", ylabel=benchmark)
    savefig(fig, savename=Path(__file__).parent / f"num_features-{benchmark}" + ("-layerwise" if per_layer else ""))


def Pereira_language_vs_other(best_layer=True):
    scores = collect_scores(benchmark='Pereira2018-encoding', models=models)
    scores_lang, scores_other = scores[scores['atlas'] == 'language'], scores[scores['atlas'] == 'auditory']
    scores_lang, scores_other = average_adjacent(scores_lang), average_adjacent(scores_other)
    scores_lang, scores_other = scores_lang.dropna(), scores_other.dropna()
    if best_layer:
        scores_lang, scores_other = choose_best_scores(scores_lang), choose_best_scores(scores_other)
    scores_lang, scores_other = align_scores(scores_lang, scores_other)
    # plot
    colors = [model_colors[model] for model in scores_lang['model'].values]
    fig, ax = _plot_scores1_2(scores_lang, scores_other, color=colors,
                              xlabel='language scores', ylabel='auditory scores')
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='dashed', color='darkgray')
    savefig(fig, savename=Path(__file__).parent / f"Pereira2018-language_other{'' if best_layer else '-layerwise'}")


def average_adjacent(data, keep_columns=('benchmark', 'model', 'layer'), skipna=False):
    data = data.groupby(list(keep_columns)).agg(lambda g: g.mean(skipna=skipna))  # mean across non-keep columns
    return data.reset_index()


def get_score_center_err(s, combine_layers=True):
    s = aggregate(s, combine_layers=combine_layers)
    if hasattr(s, 'aggregation'):
        center = s.sel(aggregation='center').values.tolist()
        try:
            error = s.sel(aggregation='error').values.tolist()
        except KeyError:
            error = np.nan
        return center, error
    if hasattr(s, 'measure'):
        if len(s['measure'].values.shape) > 0:
            if 'test_loss' in s['measure'].values:  # wikitext
                s = s.sel(measure='test_loss')
            elif 'acc_and_f1' in s['measure'].values:  # glue with acc,f1,acc_and_f1
                s = s.sel(measure='acc')
            elif 'pearson' in s['measure'].values:  # glue with pearson,spearman,corr
                s = s.sel(measure='pearson')
        s = s.values.tolist()
        return s, np.nan
    if isinstance(s, (int, float)):
        return s, np.nan
    raise ValueError(f"Unknown score structure: {s}")


def get_ceiling(benchmark, which='error', normalize_scale=True):
    """
    :param which: one of (ceiling|error|both)
    """
    if benchmark.startswith('overall'):
        metric = benchmark.split('-')[-1]
        ceilings = [get_ceiling(f"{part}-{metric}", which=which, normalize_scale=normalize_scale) for part in
                    (overall_neural_benchmarks if benchmark.startswith('overall_neural') else overall_benchmarks)]
        return np.mean(ceilings, axis=0)
    elif any(benchmark.startswith(performance_benchmark) for performance_benchmark in ['wikitext', 'glue']):
        return np.nan
    else:
        ceiling = benchmark_pool[benchmark].ceiling
        ceiling_value = ceiling.sel(aggregation='center').values
        error = ceiling.sel(aggregation=['error_low', 'error_high']).values
        if normalize_scale:  # normalize error by ceiling scale factor
            error /= ceiling_value
        output = ceiling_value if which == 'ceiling' else error if which == 'error' else (ceiling_value, error)
        return output


def shaded_errorbar(x, y, error, ax=None, shaded_kwargs=None, vertical=False, **kwargs):
    if (len(np.array(y).shape) == 1 and not is_iterable(error)) \
            or len(np.array(error).shape) == 1:  # symmetric error (only single vector)
        error_low, error_high = error, error
    else:  # asymmetric error
        assert len(error) == 2
        error = np.vectorize(lambda e: max(e, 0))(error)  # guard against negative values
        error_low, error_high = error
    shaded_kwargs = shaded_kwargs or {}
    shaded_kwargs = {**dict(linewidth=0.0), **shaded_kwargs}
    ax = ax or pyplot.gca()
    line = ax.plot(x, y, **kwargs)
    if not vertical:
        ax.fill_between(x, y - error_low, y + error_high, **shaded_kwargs)
    else:
        ax.fill_betweenx(y, x - error_low, x + error_high, **shaded_kwargs)
    return line


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for ignore_logger in ['matplotlib']:
        logging.getLogger(ignore_logger).setLevel(logging.INFO)
    seaborn.set(context='talk')
    fire.Fire()
