import numpy as np

from neural_nlp.models.implementations import model_mappings


def test_word2vec():
    model = model_mappings['word2vec']()
    _test_model(model)


def test_glove():
    model = model_mappings['glove']()
    _test_model(model)


def _test_model(model):
    encoding = model(['The quick brown fox jumps over the lazy dog'])
    assert isinstance(encoding, np.ndarray)
    print(encoding.shape)
    assert len(encoding.shape) == 2
    assert encoding.shape[0] == 1
