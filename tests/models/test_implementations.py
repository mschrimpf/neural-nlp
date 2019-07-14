import numpy as np
import pytest

from brainio_base.assemblies import NeuroidAssembly
from neural_nlp.models.implementations import load_model


class TestActivations:
    @pytest.mark.parametrize("model, num_layers", [
        ('word2vec', 1),
        ('glove', 1),
        ('lm_1b', 2),
        ('transformer', 2 * 6),
        ('bert', 12),
    ])
    def test_story_model(self, model, num_layers):
        sentence = 'The quick brown fox jumps over the lazy dog'
        model = load_model(model)
        activations = model([sentence], model.default_layers)
        assert isinstance(activations, NeuroidAssembly)
        assert 2 == len(activations.shape)
        assert num_layers == len(np.unique(activations['layer']))
        assert 1 == activations['stimulus_sentence'].shape[0]


def test_rntn():
    _test_model('rntn', sentence='If you were to journey to the North of England '
                                 'you would come to a valley that is surrounded by moors as high as')


def test_decaNLP():
    _test_model('decaNLP')


def _test_model(model_name, sentence='The quick brown fox jumps over the lazy dog'):
    model = load_model(model_name)
    encoding = model([sentence])
    assert isinstance(encoding, np.ndarray)
    print(encoding.shape)
    assert len(encoding.shape) == 2
    assert encoding.shape[0] == 1
