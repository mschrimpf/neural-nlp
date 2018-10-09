import os

import fire
import matlab.engine
from scipy.io import savemat

from brainscore.assemblies import merge_data_arrays
from neural_nlp import run

ORDERINGS = {
    'language': [32, 40, 22, 34, 35, 6, 3, 16, 11, 30, 33, 7],
    'multiple_demand': [38, 42, 28, 17, 41, 14, 8, 5, 29, 21, 25, 37, 31, 27, 26, 43, 19, 44, 15, 1],
    'default_mode_network': [36, 23, 2, 4, 18, 24, 39, 13, 9, 20, 10, 12]
}


def plot_on_brain(model, use_matlab_engine=False):
    # run & merge
    stories = ['Boar', 'KingOfBirds', 'Elvis', 'HighSchool']
    stories = [f'naturalistic-neural-reduced.{story}' for story in stories]
    scores = [run(model, story) for story in stories]
    scores = [score.aggregation.expand_dims('story') for score in scores]
    for score, story in zip(scores, stories):
        score['story'] = [story]
    scores = merge_data_arrays(scores)

    # average across stories
    means = scores.mean('story')

    # order & write out
    _matlab_engine = None
    config = {'minMax': [0., means.max().values.tolist()],
              'theMap': 'jet',
              'colorWeight': 0.25,
              'measureName': 'pearson'}
    struct_fields = {'lang': 'language', 'md': 'multiple_demand', 'dmn': 'default_mode_network'}
    for layer in scores['layer'].values:
        performance_data = {}
        for field, long_field in struct_fields.items():
            # TODO: check significance or so, don't just ignore errors
            values = means.sel(layer=layer, region=ORDERINGS[long_field], aggregation='center').values.tolist()
            performance_data[field] = values

        if not use_matlab_engine:
            savemat(f'{model}-{layer}.mat', {'perfData': performance_data, 'config': config})  # buffer for matlab
        else:
            if _matlab_engine is None:
                _matlab_engine = matlab.engine.start_matlab()
                _matlab_engine.addpath(os.path.join(os.path.dirname(__file__), '..', '..'))
            _matlab_engine.plotPerformanceOnBrain(
                {key: matlab.double(values) for key, values in performance_data.items()},
                {key: matlab.double(values) if key == 'minMax' else values
                 for key, values in config.items()})


if __name__ == '__main__':
    fire.Fire()
