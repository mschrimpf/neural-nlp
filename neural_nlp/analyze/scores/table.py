import fire
import numpy as np
from matplotlib import pyplot
from pathlib import Path

from neural_nlp import score


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


if __name__ == '__main__':
    fire.Fire()
