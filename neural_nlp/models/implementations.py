import logging
import os

import numpy as np

_ressources_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ressources', 'models')


class Model(object):
    pass


class SkipThoughts(Model):
    """
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """

    def __init__(self, weights=os.path.join(_ressources_dir, 'skip-thoughts')):
        import skipthoughts
        weights = weights + '/'
        model = skipthoughts.load_model(path_to_models=weights, path_to_tables=weights)
        self._encoder = skipthoughts.Encoder(model)

    def __call__(self, sentences):
        return self._encoder.encode(sentences)


class LM1B(Model):
    """
    https://arxiv.org/pdf/1602.02410.pdf
    """

    def __init__(self, weights=os.path.join(_ressources_dir, 'lm_1b')):
        from lm_1b.lm_1b_eval import Encoder
        self._encoder = Encoder(vocab_file=os.path.join(weights, 'vocab-2016-09-10.txt'),
                                pbtxt=os.path.join(weights, 'graph-2016-09-10.pbtxt'),
                                ckpt=os.path.join(weights, 'ckpt-*'))

    def __call__(self, sentences):
        embeddings, word_ids = self._encoder(sentences)
        return np.array([embedding[-1][0] for embedding in embeddings])  # only output last embedding, discard time


class KeyedVectorModel(Model):
    def __init__(self, weights_file, binary=False):
        from gensim.models.keyedvectors import KeyedVectors
        self._model = KeyedVectors.load_word2vec_format(weights_file, binary=binary)
        self._index2word_set = set(self._model.index2word)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _combine_vectors(self, feature_vectors):
        return _mean_vector(feature_vectors)

    def __call__(self, sentences):
        return np.array([self._encode_sentence(sentence) for sentence in sentences])

    def _encode_sentence(self, sentence):
        words = sentence.split()
        feature_vectors = []
        for word in words:
            if word in self._index2word_set:
                feature_vectors.append(self._model[word])
            else:
                self._logger.warning("Word {} not present in model".format(word))
        return self._combine_vectors(feature_vectors)


class Word2Vec(KeyedVectorModel):
    """
    https://arxiv.org/pdf/1310.4546.pdf
    """

    def __init__(self, weights_file='GoogleNews-vectors-negative300.bin'):
        weights_file = os.path.join(_ressources_dir, 'word2vec', weights_file)
        super(Word2Vec, self).__init__(weights_file=weights_file, binary=True)


class Glove(KeyedVectorModel):
    """
    http://www.aclweb.org/anthology/D14-1162
    """

    def __init__(self, weights='glove.840B.300d.txt'):
        from gensim.scripts.glove2word2vec import glove2word2vec
        weights_file = os.path.join(_ressources_dir, 'glove', weights)
        word2vec_weightsfile = weights_file + '.word2vec'
        if not os.path.isfile(word2vec_weightsfile):
            glove2word2vec(weights_file, word2vec_weightsfile)
        super(Glove, self).__init__(weights_file=word2vec_weightsfile)


class RecursiveNeuralTensorNetwork(Model):
    """
    http://www.aclweb.org/anthology/D13-1170
    """
    pass


def _mean_vector(feature_vectors):
    num_words = len(feature_vectors)
    assert num_words > 0
    if num_words == 1:
        return feature_vectors[0]
    feature_vectors = np.sum(feature_vectors, axis=0)
    return np.divide(feature_vectors, num_words)


def load_model(model_name):
    return _model_mappings[model_name]()


_model_mappings = {
    'skip-thoughts': SkipThoughts,
    'lm_1b': LM1B,
    'word2vec': Word2Vec,
    'glove': Glove
}
