from neural_nlp.models.implementations import word2vec_mean
import numpy as np


def test_word2vec():
    model = word2vec_mean()
    encoding = model('The quick brown fox jumps over the lazy dog')
    assert isinstance(encoding, np.ndarray)
    assert len(encoding.shape) == 1
