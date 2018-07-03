import itertools

import numpy as np
import pytest

from neural_nlp import run


class TestRun:
    models = ['rntn', 'skip-thoughts']
    stories = ['Boar', 'MatchstickSeller', 'KingOfBirds', 'Elvis', 'HighSchool']

    @pytest.mark.parametrize("model, stimulus_set", list(itertools.product(models, stories)))
    def test_story_model(self, model, stimulus_set):
        scores = run(model=model, stimulus_set='naturalistic-neural-reduced.{}'.format(stimulus_set))
        assert 'region' in scores.aggregation
        np.testing.assert_array_equal(scores.aggregation.sel(aggregation='center').shape, [44])
