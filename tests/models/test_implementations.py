import numpy as np

from neural_nlp.models.implementations import load_model


def test_word2vec():
    _test_model('word2vec')


def test_glove():
    _test_model('glove')


def test_rntn():
    _test_model('rntn', sentence='If you were to journey to the North of England, '
                                 'you would come to a valley that is surrounded by moors as high as mountains.')


def test_decaNLP():
    _test_model('decaNLP')
    
    
def _test_model(model_name, sentence='The quick brown fox jumps over the lazy dog'):
    model = load_model(model_name)
    encoding = model([sentence])
    assert isinstance(encoding, np.ndarray)
    print(encoding.shape)
    assert len(encoding.shape) == 2
    assert encoding.shape[0] == 1
