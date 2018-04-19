import numpy as np

from neural_nlp import run


class TestRun:
    def test_boar_skip_thoughts(self):
        region_scores = run(model='skip-thoughts', stimulus_set='naturalistic-neural-reduced.Boar')
        assert 'region' in region_scores
        np.testing.assert_array_equal(region_scores.shape, [44])
