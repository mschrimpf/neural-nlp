import numpy as np

from brainscore.assemblies import NeuroidAssembly
from neural_nlp.models.implementations import load_model


def test_lm1b():
    sentence = 'The quick brown fox jumps over the lazy dog'
    model = load_model('lm_1b')
    activations = model.get_activations([sentence], model.default_layers())
    assert isinstance(activations, NeuroidAssembly)
    assert 2 == len(activations.shape)
    assert 2 == len(np.unique(activations['layer']))
    assert 1 == activations['sentence'].shape[0]


def test_word2vec():
    _test_model('word2vec')


def test_glove():
    _test_model('glove')


def test_rntn():
    _test_model('rntn', sentence='If you were to journey to the North of England '
                                 'you would come to a valley that is surrounded by moors as high as')


def _test_model(model_name, sentence='The quick brown fox jumps over the lazy dog'):
    model = load_model(model_name)
    encoding = model([sentence])
    assert isinstance(encoding, np.ndarray)
    print(encoding.shape)
    assert len(encoding.shape) == 2
    assert encoding.shape[0] == 1
