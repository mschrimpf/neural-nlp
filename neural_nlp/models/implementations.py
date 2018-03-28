import os

import gensim.models.keyedvectors as word2vec
import numpy as np

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


def word2vec_mean(weights=os.path.join(_ressources_dir, 'word2vec', 'GoogleNews-vectors-negative300.bin')):
    """
    https://arxiv.org/pdf/1310.4546.pdf
    """
    model = word2vec.KeyedVectors.load_word2vec_format(weights, binary=True)
    index2word_set = set(model.index2word)

    def mean_feature_vector(sentence):
        words = sentence.split()
        feature_vec = None
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word]) if feature_vec is not None else model[word]
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    return mean_feature_vector


model_mappings = {
    'skip-thoughts': skip_thoughts,
    'lm_1b': lm_1b,
    'word2vec': word2vec_mean
}
