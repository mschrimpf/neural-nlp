import numpy as np

from neural_nlp.models.implementations import word2vec, glove


def test_word2vec():
    model = word2vec()
    _test_model(model)


def test_glove():
    model = glove()
    _test_model(model)


def _test_model(model):
    encoding = model('The quick brown fox jumps over the lazy dog')
    assert isinstance(encoding, np.ndarray)
    assert len(encoding.shape) == 1
