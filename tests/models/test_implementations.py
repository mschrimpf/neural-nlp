import numpy as np

from neural_nlp.models.implementations import load_model


def test_word2vec():
    _test_model('word2vec')


def test_glove():
    _test_model('glove')


def _test_model(model_name):
    model = load_model(model_name)
    encoding = model(['The quick brown fox jumps over the lazy dog'])
    assert isinstance(encoding, np.ndarray)
    print(encoding.shape)
    assert len(encoding.shape) == 2
    assert encoding.shape[0] == 1
