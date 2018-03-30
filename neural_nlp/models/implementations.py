import os

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

_ressources_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ressources', 'models')


def skip_thoughts(weights=os.path.join(_ressources_dir, 'skip-thoughts')):
    """
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """
    import skipthoughts
    model = skipthoughts.load_model(path_to_models=weights, path_to_tables=weights)
    encoder = skipthoughts.Encoder(model)
    return encoder.encode


def lm_1b(weights=os.path.join(_ressources_dir, 'lm_1b')):
    """
    https://arxiv.org/pdf/1602.02410.pdf
    """
    from lm_1b.lm_1b_eval import Encoder

    encoder = Encoder(vocab_file=os.path.join(weights, 'vocab-2016-09-10.txt'),
                      pbtxt=os.path.join(weights, 'graph-2016-09-10.pbtxt'),
                      ckpt=os.path.join(weights, 'ckpt-*'))

    def encode(sentences):
        embeddings, word_ids = encoder(sentences)
        return np.array([embedding[-1][0] for embedding in embeddings])  # only output last embedding, discard time

    return encode


def _mean_vector(feature_vectors, num_words):
    assert len(feature_vectors) == num_words
    assert num_words > 0
    if num_words == 1:
        return feature_vectors[0]
    feature_vectors = np.sum(feature_vectors, axis=0)
    return np.divide(feature_vectors, num_words)


def _keyed_vector_model(weights, combine_vectors, binary=False):
    model = KeyedVectors.load_word2vec_format(weights, binary=binary)
    index2word_set = set(model.index2word)

    def combined_feature_vector(sentence):
        words = sentence.split()
        feature_vectors = []
        num_words = 0
        for word in words:
            if word in index2word_set:
                num_words += 1
                feature_vectors.append(model[word])
        return combine_vectors(feature_vectors, num_words)

    return combined_feature_vector


def word2vec(weights='GoogleNews-vectors-negative300.bin', combine_vectors=_mean_vector):
    """
    https://arxiv.org/pdf/1310.4546.pdf
    """
    weights_file = os.path.join(_ressources_dir, 'word2vec', weights)
    return _keyed_vector_model(weights_file, combine_vectors=combine_vectors, binary=True)


def glove(weights='glove.840B.300d.txt', combine_vectors=_mean_vector):
    """
    http://www.aclweb.org/anthology/D14-1162
    """
    weights_file = os.path.join(_ressources_dir, 'glove', weights)
    word2vec_weightsfile = weights_file + '.word2vec'
    if not os.path.isfile(word2vec_weightsfile):
        glove2word2vec(weights_file, word2vec_weightsfile)
    return _keyed_vector_model(word2vec_weightsfile, combine_vectors=combine_vectors)


model_mappings = {
    'skip-thoughts': skip_thoughts,
    'lm_1b': lm_1b,
    'word2vec': word2vec,
    'glove': glove
}
