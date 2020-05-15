import fire
import numpy as np
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import score
from neural_nlp.analyze.scores import collect_scores, average_adjacent, models, choose_best_scores


def render_mpl_table(layer_scores, col_width=0.3, row_height=0.5, font_size=14,
                     header_color='#40466e', row_colors=('#f1f1f2', 'w'), edge_color='w',
                     bbox=(0.25, 0, 0.75, 1), header_columns=0,
                     ax=None, **kwargs):
    assert set(layer_scores.dims) == {'layer', 'atlas'}
    layer_scores = layer_scores.transpose('layer', 'atlas')
    atlas_ordering = ['DMN', 'MD', 'language', 'auditory', 'visual']
    assert set(layer_scores['atlas'].values) == set(atlas_ordering)
    layer_scores = layer_scores[{'atlas': [atlas_ordering.index(atlas) for atlas in layer_scores['atlas'].values]}]
    if ax is None:
        size = (np.array(layer_scores.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height])
        fig, ax = pyplot.subplots(figsize=size)
        ax.axis('off')

    cellText = np.array([f"{value:.2f}" for value in layer_scores.values.flatten()]).reshape(layer_scores.shape)
    mpl_table = ax.table(cellText=cellText, bbox=bbox, cellLoc='center',
                         rowLabels=layer_scores['layer'].values, colLabels=layer_scores['atlas'].values, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        # coloring/font
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        # width
        if k[1] >= 0:
            cell.set_width(0.14)
    return ax


def scores_per_region_and_model(models=('bert', 'bert-untrained'), benchmark='Pereira2018-encoding'):
    for model in models:
        model_scores = score(benchmark=benchmark, model=model)
        model_scores = model_scores.sel(aggregation='center').mean('experiment')

        render_mpl_table(model_scores, header_columns=0, col_width=2.0)
        pyplot.tight_layout()
        pyplot.savefig(Path(__file__).parent / f"table-regions-{model}.png")


def overview_table():
    benchmarks = {'Pereira2018': 'Pereira2018-encoding', 'Fedorenko2016': 'Fedorenko2016v3-encoding',
                  'Blank2014': 'Blank2014fROI-encoding', 'Futrell2018': 'Futrell2018-encoding',
                  'Average': 'overall-encoding'}

    rows = [{**dict(header=""), **{name: name for name in benchmarks}},
            {**dict(header=''), **{name: ''.join(['-'] * 40) for name in benchmarks}}]
    # pre-run to ensure all models are in csv
    all_models = [[model, f"{model}-untrained"] for model in models]
    all_models = [model for model_tuple in all_models for model in model_tuple]
    _max_benchmark_scores(benchmarks, models=all_models)
    # max
    row = dict(header="Maximal Predictivity")
    score_row = _max_benchmark_scores(benchmarks, models=models)
    rows.append({**row, **score_row})
    # untrained
    row = dict(header="- Architecture only (no training)")
    untrained_models = [f"{model}-untrained" for model in models]
    score_row = _max_benchmark_scores(benchmarks, models=untrained_models)
    rows.append({**row, **score_row})

    # output
    for row in rows:
        s = f"{row['header']: <40} | " + " | ".join(f"{row[name]: >40}" for name in benchmarks)
        print(s)


def _max_benchmark_scores(benchmarks, models):
    row = {}
    for name, benchmark in benchmarks.items():
        scores = _model_scores(benchmark, models=models)
        max_score = scores.ix[scores['score'].idxmax()]
        row[name] = f"{100 * min(1, max_score['score']):.0f}% ({max_score['model']})"
    return row


def _model_scores(benchmark, models):
    scores = collect_scores(benchmark=benchmark, models=models, normalize=True)
    scores = average_adjacent(scores).dropna()
    scores = choose_best_scores(scores)
    return scores


if __name__ == '__main__':
    fire.Fire()
