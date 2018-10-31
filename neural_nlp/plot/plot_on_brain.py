from collections import defaultdict

import fire
import numpy as np
import os
from scipy.io import savemat

from neural_nlp import STORIES, run_stories
from neural_nlp.plot.stats import is_significant

REGION_ORDERINGS = {
    'language': [32, 40, 22, 34, 35, 6, 3, 16, 11, 30, 33, 7],
    'multiple_demand': [38, 42, 28, 17, 41, 14, 8, 5, 29, 21, 25, 37, 31, 27, 26, 43, 19, 44, 15, 1],
    'default_mode_network': [36, 23, 2, 4, 18, 24, 39, 13, 9, 20, 10, 12]
}
REGION_FIELDS = {'lang': 'language', 'md': 'multiple_demand', 'dmn': 'default_mode_network'}


def plot_preferences(model, use_matlab_engine=False, stories=STORIES):
    """
    plot which layer most prefers a region
    """
    scores = run_stories(model=model, stories=stories)

    # average across stories
    scores = scores.mean('story')

    # order & write out
    _matlab_engine = None
    config = {'minMax': [0., 1.],
              'theMap': 'winter',
              'colorWeight': 0.25,
              'measureName': 'layer-preference'}
    performance_data = defaultdict(list)
    for field, long_field in REGION_FIELDS.items():
        def best_layer(group):
            argmax = group.sel(aggregation='center').argmax('layer')
            return group[:, argmax.values]

        max_score = scores.groupby('region').apply(best_layer)
        region_layers = [scores['layer'][(scores == max_score).sel(aggregation='center', region=region)].values[0]
                         for region in REGION_ORDERINGS[long_field]]
        positions = [relative_position(layer, scores['layer'].values.tolist()) for layer in region_layers]
        performance_data[field] = positions

    savepath = f'{model}.mat'
    savemat(savepath, {'perfData': performance_data, 'config': config})  # buffer for matlab
    if use_matlab_engine:
        import matlab.engine
        if _matlab_engine is None:
            _matlab_engine = matlab.engine.start_matlab()
            _matlab_engine.addpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        _matlab_engine.plotPerformanceOnBrain(savepath)


def plot_layerwise(model, use_matlab_engine=False, stories=STORIES):
    """
    plot scores of each layer separately
    """
    scores = run_stories(model=model, stories=stories)
    reference_scores = run_stories(model='random-gaussian', stories=stories).squeeze('layer')

    # average across stories
    scores = scores.mean('story', _apply_raw=True)
    reference_scores = reference_scores.mean('story', _apply_raw=True)

    # order & write out
    _matlab_engine = None
    config = {'minMax': [0., scores.max().values.tolist()],
              'theMap': 'jet',
              'colorWeight': 0.25,
              'measureName': 'pearson'}
    for layer in scores['layer'].values:
        performance_data = {}
        for region_short, region_name in REGION_FIELDS.items():
            region_ids = REGION_ORDERINGS[region_name]
            layer_values = scores.sel(layer=layer, region=region_ids)
            significant = [is_significant(scores=layer_values.sel(region=region).raw.values,
                                          reference_scores=reference_scores.sel(region=region).raw.values)
                           for region in region_ids]
            values = layer_values.sel(aggregation='center', _apply_raw=False).values
            values[np.invert(significant)] = -1
            performance_data[region_short] = values.tolist()

        savepath = f'{model}-{layer}.mat'
        savemat(savepath, {'perfData': performance_data, 'config': config})  # buffer for matlab
        if use_matlab_engine:
            if _matlab_engine is None:
                import matlab.engine
                _matlab_engine = matlab.engine.start_matlab()
                _matlab_engine.addpath(os.path.join(os.path.dirname(__file__), '..', '..'))
            _matlab_engine.plotPerformanceOnBrain(savepath)


def relative_position(elem, collection):
    return collection.index(elem) / len(collection)


if __name__ == '__main__':
    fire.Fire()
