import logging
import os

import numpy as np
import pandas as pd

from neural_nlp.models.wrapper import DeepModel

_ressources_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ressources', 'models')


class Model(object):
    def __call__(self, sentences):
        raise NotImplementedError()


class SkipThoughts(DeepModel):
    """
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """

    def __init__(self, weights=os.path.join(_ressources_dir, 'skip-thoughts')):
        super().__init__()
        import skipthoughts
        weights = weights + '/'
        model = skipthoughts.load_model(path_to_models=weights, path_to_tables=weights)
        self._encoder = skipthoughts.Encoder(model)

    def _get_activations(self, sentences, layer_names):
        np.testing.assert_array_equal(layer_names, ['encoder'])
        encoding = self._encoder.encode(sentences)
        return {'encoder': encoding}

    @classmethod
    def available_layers(cls):
        return ['encoder']

    @classmethod
    def default_layers(cls):
        return cls.available_layers()


class LM1B(DeepModel):
    """
    https://arxiv.org/pdf/1602.02410.pdf
    """

    def __init__(self, weights=os.path.join(_ressources_dir, 'lm_1b')):
        super().__init__()
        from lm_1b.lm_1b_eval import Encoder
        self._encoder = Encoder(vocab_file=os.path.join(weights, 'vocab-2016-09-10.txt'),
                                pbtxt=os.path.join(weights, 'graph-2016-09-10.pbtxt'),
                                ckpt=os.path.join(weights, 'ckpt-*'))

    def _get_activations(self, sentences, layer_names):
        from lm_1b import lm_1b_eval
        from six.moves import xrange
        # the following is copied from lm_1b.lm_1b_eval.Encoder.__call__.
        # only the `sess.run` call needs to be changed but there's no way to access it outside the code
        self._encoder.sess.run(self._encoder.t['states_init'])
        targets = np.zeros([lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS], np.int32)
        weights = np.ones([lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS], np.float32)
        sentences_embeddings, sentences_word_ids = [], []
        for sentence in sentences:
            if sentence.find('<S>') != 0:
                sentence = '<S> ' + sentence
            word_ids = [self._encoder.vocab.word_to_id(w) for w in sentence.split()]
            char_ids = [self._encoder.vocab.word_to_char_ids(w) for w in sentence.split()]
            inputs = np.zeros([lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS], np.int32)
            char_ids_inputs = np.zeros(
                [lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS, self._encoder.vocab.max_word_length], np.int32)
            embeddings = []
            for i in xrange(len(word_ids)):
                inputs[0, 0] = word_ids[i]
                char_ids_inputs[0, 0, :] = char_ids[i]

                # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
                # LSTM.
                lstm_emb = self._encoder.sess.run([self._encoder.t[name] for name in layer_names],
                                                  feed_dict={self._encoder.t['char_inputs_in']: char_ids_inputs,
                                                             self._encoder.t['inputs_in']: inputs,
                                                             self._encoder.t['targets_in']: targets,
                                                             self._encoder.t['target_weights_in']: weights})
                embeddings.append(lstm_emb)
            sentences_embeddings.append(embeddings)
            sentences_word_ids.append(word_ids)
        # `sentences_embeddings` shape is now: sentences x words x layers x *layer_shapes
        layer_activations = {}
        for i, layer in enumerate(layer_names):
            # only output last embedding (last word), discard time course
            layer_activations[layer] = np.array([embedding[-1][i] for embedding in sentences_embeddings])
        return layer_activations

    def available_layers(self, filter_inputs=True):
        return [tensor_name for tensor_name in self._encoder.t if not filter_inputs or not tensor_name.endswith('_in')]

    @classmethod
    def default_layers(cls):
        return ['lstm/lstm_0/control_dependency', 'lstm/lstm_1/control_dependency']


class KeyedVectorModel(DeepModel):
    """
    Lookup-table-like models where each word has an embedding.
    To retrieve the sentence activation, we take the mean of the word embeddings.
    """

    def __init__(self, weights_file, binary=False):
        from gensim.models.keyedvectors import KeyedVectors
        self._model = KeyedVectors.load_word2vec_format(weights_file, binary=binary)
        self._index2word_set = set(self._model.index2word)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _combine_vectors(self, feature_vectors):
        return _mean_vector(feature_vectors)

    def _get_activations(self, sentences, layer_names):
        np.testing.assert_array_equal(layer_names, ['projection'])
        encoding = np.array([self._encode_sentence(sentence) for sentence in sentences])
        return {'projection': encoding}

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

    @classmethod
    def available_layers(cls):
        return ['projection']

    @classmethod
    def default_layers(cls):
        return cls.available_layers()


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

    def __init__(self, weights='sentiment'):
        cachepath = os.path.join(_ressources_dir, 'recursive-neural-tensor-network', weights + '.activations.csv')
        self._cache = pd.read_csv(cachepath)
        self._cache = self._cache[self._cache['node.type'] == 'ROOT']
        self._cache.drop_duplicates(inplace=True)

    def __call__(self, sentences):
        result = self._cache[self._cache['sentence'].isin(sentences)]
        assert len(result) == 1
        result = result[[column for column in result if column.startswith('activation')]]
        return result.values


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
    'glove': Glove,
    'rntn': RecursiveNeuralTensorNetwork,
}

model_layers = {
    'skip-thoughts': SkipThoughts.default_layers(),
    'lm_1b': LM1B.default_layers(),
    'word2vec': Word2Vec.default_layers(),
    'glove': Glove.default_layers(),
}
