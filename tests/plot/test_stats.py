import numpy as np

from neural_nlp import run_stories
from neural_nlp.plot.stats import is_significant


class TestIsSignificant:
    def test_random(self):
        region = 1
        scores = run_stories(model='random-gaussian').sel(region=region).mean('story', _apply_raw=True).squeeze('layer')
        significant = is_significant(scores=scores.raw.values, reference_scores=scores.raw.values)
        assert not significant

    def test_insignificant(self):
        scores = [94, 197, 16, 38, 99, 141, 23]
        reference_scores = [52, 104, 146, 10, 51, 30, 40, 27, 46]
        significant = is_significant(scores=scores, reference_scores=reference_scores)
        assert not significant

    def test_significant(self):
        scores = np.random.normal(loc=5, size=10)
        reference_scores = np.random.normal(loc=0, size=10)
        significant = is_significant(scores=scores, reference_scores=reference_scores)
        assert significant
