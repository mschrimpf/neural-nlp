import numpy as np
from matplotlib import pyplot, gridspec

from neural_nlp import run

# left hemisphere language network + default mode network
region_selection = [32, 40, 22, 34, 35, 6] + [36, 23, 2, 4, 18, 24, 39, 13, 9, 20, 10, 12]


def plot_similarities(model, story, fig=None, grid_spec=None, show_region_text=False, title_prefix=None):
    score = run(model, 'naturalistic-neural-reduced.{}'.format(story))
    score = score.aggregation

    regions = score['region'].values
    layers = np.unique(score['layer'])
    if grid_spec is None:
        assert fig is None
        fig, axes = pyplot.subplots(len(layers), 1, figsize=(12, 12))
    else:
        assert fig is not None
        grid = gridspec.GridSpecFromSubplotSpec(len(layers), 1, subplot_spec=grid_spec, wspace=.1)
        axes = []
        for i in range(len(layers)):
            ax = pyplot.Subplot(fig, grid[i])
            axes.append(ax)
            fig.add_subplot(ax)

    for ax, layer in zip(axes, layers):
        layer_score = score.sel(layer=layer)
        y, yerr = layer_score.sel(aggregation='center'), layer_score.sel(aggregation='error')

        ax.set_title("{}Layer {}".format((title_prefix + " ") if title_prefix is not None else "", layer))
        bar = ax.bar(regions, y, yerr=yerr)
        ax.errorbar(regions, y, yerr=yerr, linestyle='None', color='gray', elinewidth=1)
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        if show_region_text:
            for rect, region in zip(bar.patches, regions):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        region, ha='center', va='bottom')
    return fig
