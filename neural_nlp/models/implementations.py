import copy
import logging
import os
import tempfile
from collections import OrderedDict
from enum import Enum
from importlib import import_module

import itertools
import numpy as np
import pandas as pd
from numpy.random import RandomState
from tqdm import tqdm

import pickle

from brainscore.utils import LazyLoad
from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
from neural_nlp.models.wrapper.pytorch import PytorchWrapper

_ressources_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'ressources', 'models')

###### for testing replace "__file__" with "os.getcwd()"


class BrainModel:
    Modes = Enum('Mode', 'recording general_features')

    def __init__(self):
        super(BrainModel, self).__init__()
        self._mode = self.Modes.recording

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in self.Modes
        self._mode = value

    def __call__(self, sentences):
        raise NotImplementedError()


class GaussianRandom:
    _layer_name = 'random'
    available_layers = [_layer_name]
    default_layers = available_layers

    def __init__(self, num_samples=1000):
        """
        :param num_samples: how many samples to draw for each sentence
        """
        self._rng = RandomState()
        self._num_samples = num_samples
        super(GaussianRandom, self).__init__()

    def _get_activations(self, sentences, layer_names):
        assert layer_names == [self._layer_name]
        return {self._layer_name: self._rng.standard_normal((len(sentences), self._num_samples))}

    
    
class TopicETM:
    """https://arxiv.org/abs/1907.04907"""

    def __init__(self):
        weights_file = os.path.join(_ressources_dir, 'normalized_betas.npy')
        vocab_file = os.path.join(_ressources_dir, 'vocab.pkl')
        
        super().__init__()
        
        self.weights = np.load(weights_file)
        with open(vocab_file,'rb') as f:
             self.vocab = pickle.load(f)
        
        wordEmb_TopicSpace = {}
        for elm in tqdm(self.vocab, desc='vocab'):
            i = self.vocab.index(elm) # get index of word
            wordEmb_TopicSpace[elm] = self.weights[:,i]
        self.wordEmb_TopicSpace = wordEmb_TopicSpace
        self._extractor = ActivationsExtractorHelper(identifier='topicETM', get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)
        self._extractor.register_activations_hook(word_mean)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)       
        
    def _encode_sentence(self, sentence):
        words = sentence.split()
        feature_vectors = []        
                 
        for word in words:
            if word in self.vocab:
                feature_vectors.append(self.wordEmb_TopicSpace[word])
            else:
                self._logger.warning(f"Word {word} not present in model")
                #feature_vectors.append(np.zeros([1,50]))
        #### BREAK & check Shape
        #print(feature_vectors)
        #feature_vectors = [np.expand_dims(sentence_encodings, 0) for sentence_encodings in feature_vectors]

        #sentence_enc = np.mean(feature_vectors, axis=0)
        return feature_vectors
    
    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['projection'])
        encoding = [np.array(self._encode_sentence(sentence)) for sentence in sentences]
        return {'projection': encoding}
                
    available_layers = ['projection']
    default_layers = available_layers

    
class SkipThoughts:
    """
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """

    def __init__(self, weights=os.path.join(_ressources_dir, 'skip-thoughts')):
        super().__init__()
        import skipthoughts
        weights = weights + '/'
        model = LazyLoad(lambda: skipthoughts.load_model(path_to_models=weights, path_to_tables=weights))
        self._encoder = LazyLoad(lambda: skipthoughts.Encoder(model))
        self._extractor = ActivationsExtractorHelper(identifier='skip-thoughts', get_activations=self._get_activations,
                                                     reset=lambda: None)  # TODO: no idea how to reset state in theano.
        self._extractor.insert_attrs(self)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['encoder'])
        encoding = self._encoder.encode(sentences)
        return {'encoder': encoding}

    available_layers = ['encoder']
    default_layers = available_layers


class LM1B:
    """
    https://arxiv.org/pdf/1602.02410.pdf
    """

    def __init__(self, weights=os.path.join(_ressources_dir, 'lm_1b')):
        super().__init__()
        from lm_1b.lm_1b_eval import Encoder
        self._encoder = Encoder(vocab_file=os.path.join(weights, 'vocab-2016-09-10.txt'),
                                pbtxt=os.path.join(weights, 'graph-2016-09-10.pbtxt'),
                                ckpt=os.path.join(weights, 'ckpt-*'))
        self._extractor = ActivationsExtractorHelper(identifier='lm_1b', get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)
        self._extractor.register_activations_hook(word_last)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def _get_activations(self, sentences, layers):
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
                # TODO: ensure this preserves hidden state
                lstm_emb = self._encoder.sess.run([self._encoder.t[name] for name in layers],
                                                  feed_dict={self._encoder.t['char_inputs_in']: char_ids_inputs,
                                                             self._encoder.t['inputs_in']: inputs,
                                                             self._encoder.t['targets_in']: targets,
                                                             self._encoder.t['target_weights_in']: weights})
                embeddings.append(lstm_emb)
            sentences_embeddings.append(embeddings)
            sentences_word_ids.append(word_ids)
        # `sentences_embeddings` shape is now: sentences x words x layers x *layer_shapes
        layer_activations = {}
        for i, layer in enumerate(layers):
            # sentences_embeddings is `sentences x words x layers x (1 x 1024)`
            layer_activations[layer] = [np.array(embedding)[:, i] for embedding in sentences_embeddings]
            # words x 1 x 1024 --> 1 x words x 1024
            layer_activations[layer] = [embedding.transpose(1, 0, 2) for embedding in layer_activations[layer]]
        return layer_activations

    def available_layers(self, filter_inputs=True):
        return [tensor_name for tensor_name in self._encoder.t if not filter_inputs or not tensor_name.endswith('_in')]

    default_layers = ['lstm/lstm_0/control_dependency', 'lstm/lstm_1/control_dependency']


def word_last(layer_activations):
    for layer, activations in layer_activations.items():
        assert all(a.shape[0] == 1 for a in activations)
        activations = [a[0, -1, :] for a in activations]
        layer_activations[layer] = np.array(activations)
    return layer_activations


def word_mean(layer_activations):
    for layer, activations in layer_activations.items():
        activations = [np.mean(a, axis=1) for a in activations]  # average across words within a sentence
        layer_activations[layer] = np.concatenate(activations)
    return layer_activations


def subsample_random(sentence_activations, num_components=1000):
    for layer, layer_activations in sentence_activations.items():
        subsampled_layer_activations = []
        for activations in layer_activations:
            activations = activations.reshape(activations.shape[0], -1)
            indices = np.random.randint(activations.shape[1], size=num_components)
            activations = activations[:, indices]
            subsampled_layer_activations.append(activations)
        sentence_activations[layer] = np.concatenate(subsampled_layer_activations)
    return sentence_activations


def pad_zero(sentence_activations):
    for layer, layer_activations in sentence_activations.items():
        per_word_features = layer_activations[0].shape[-1]
        max_num_features = max(a.shape[1] for a in layer_activations)
        max_num_features = max_num_features * per_word_features

        padded_layer_activations = []
        for activations in layer_activations:
            activations = activations.reshape(activations.shape[0], -1)
            activations = np.pad(activations, pad_width=((0, 0), (0, max_num_features - activations.size)),
                                 mode='constant', constant_values=0)
            padded_layer_activations.append(activations)
        sentence_activations[layer] = np.array(padded_layer_activations)
    return sentence_activations


def Transformer_WordAll():
    """
    use representations for all the words. Due to different sentence lengths,
    this will most likely only work with sub-sampling.
    However, even then we're subsampling at different locations, making this unlikely to yield reliable representations.
    :return:
    """
    transformer = Transformer()

    def combine_word_activations(layer_activations):
        for layer, activations in layer_activations.items():
            activations = [a.reshape(a.shape[0], -1) for a in activations]
            layer_activations[layer] = np.concatenate(activations)
        return layer_activations

    transformer.register_activations_hook(combine_word_activations)
    transformer._extractor.identifier += '-wordall'
    return transformer


def Transformer_WordLast():
    transformer = Transformer()
    transformer.register_activations_hook(word_last)
    transformer._extractor.identifier += '-wordlast'
    return transformer


def Transformer_WordMean():
    transformer = Transformer()
    transformer.register_activations_hook(word_mean)
    transformer._extractor.identifier += '-wordmean'
    return transformer


def Transformer_SubsampleRandom():
    transformer = Transformer()
    transformer.register_activations_hook(subsample_random)
    transformer._extractor.identifier += '-subsample_random'
    return transformer


def Transformer_PadZero():
    transformer = Transformer()
    transformer.register_activations_hook(pad_zero)
    transformer._extractor.identifier += '-pad_zero'
    return transformer


class Transformer(PytorchWrapper):
    """
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self):
        weights = os.path.join(_ressources_dir, 'transformer/averaged-10-epoch.pt')
        from onmt.opts import add_md_help_argument, translate_opts
        from onmt.translate.translator import build_translator
        import argparse
        parser = argparse.ArgumentParser(description='transformer-parser-base')
        add_md_help_argument(parser)
        translate_opts(parser, weights)
        opt = parser.parse_args(['-batch_size', '1'])
        translator = build_translator(opt, report_score=True)

        model_container = self.TransformerContainer(translator, opt)
        super(Transformer, self).__init__(model=model_container, identifier='transformer',
                                          reset=lambda: None)  # transformer is feed-forward

    class TransformerContainer:
        def __init__(self, translator, opt):
            self.translator = translator
            self.opt = opt

        def __getattr__(self, name):
            return getattr(self.translator.model, name)

        def __call__(self, sentences):
            with tempfile.NamedTemporaryFile(mode='w+') as file:
                # separating sentences with newline, combined with a batch size of 1
                # will lead to one set of activations per sentence (albeit multiple words).
                file.write('\n'.join(sentences) + '\n')
                file.flush()
                encodings = self.translator.get_encodings(src_path=file.name, tgt_path=self.opt.tgt,
                                                          src_dir=self.opt.src_dir, batch_size=self.opt.batch_size,
                                                          attn_debug=self.opt.attn_debug)
                return encodings

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            numpy_output = PytorchWrapper._tensor_to_numpy(output)
            target_dict[name].append(numpy_output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    default_layers = [f'encoder.transformer.{i}.{layer}'
                      for i in range(6) for layer in ['feed_forward.layer_norm', 'feed_forward.dropout_2']]
    """
    For each of the 6 encoder blocks, we're using two layers,
    one following the Multi-Head Attention and one following the Feed Forward block (cf. Figure 1).
    
    The encoder is implemented as follows:
    ```
    input_norm = self.layer_norm(inputs)
    context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
    out = self.dropout(context) + inputs
    return self.feed_forward(out)
    ```
    `feed_forward` is implemented as follows:
    ```
    inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
    output = self.dropout_2(self.w_2(inter))
    return output + x
    ```
    We thus use `feed_forward.layer_norm` as the layer immediately following the Multi-Head Attention
    and `feed_forward.dropout_2` as the last layer of the Feed Forward block.
    Note however that the attended input has not yet been added back to the feed forward output with
    `feed_forward.dropout_2`; with this framework we cannot capture that operation (we'd have to change the code).
    """


class _PytorchTransformerWrapper(BrainModel):
    def __init__(self, identifier, tokenizer, model, layers, tokenizer_special_tokens=()):
        super(_PytorchTransformerWrapper, self).__init__()
        self.default_layers = self.available_layers = layers
        self._model = model
        self._model_container = self.ModelContainer(tokenizer, model, layers, tokenizer_special_tokens)
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self._model_container,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        self._model.eval()
        if self.mode == BrainModel.Modes.recording:
            return self._extractor(*args, **kwargs)
        elif self.mode == BrainModel.Modes.general_features:
            # (input_ids, position_ids=position_ids, token_type_ids=token_type_ids, head_mask=head_mask)
            transformer_outputs = self._model(*args, **kwargs)
            hidden_states = transformer_outputs[0]
            return hidden_states
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    @property
    def identifier(self):
        return self._extractor.identifier

    @property
    def tokenizer(self):
        return self._model_container.tokenizer

    @property
    def features_size(self):
        return self._model.config.hidden_size

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    class ModelContainer:
        def __init__(self, tokenizer, model, layer_names, tokenizer_special_tokens=()):
            self.tokenizer = tokenizer
            self.model = model
            self.layer_names = layer_names
            self.tokenizer_special_tokens = tokenizer_special_tokens

        def __call__(self, sentences, layers):
            import torch
            self.model.eval()
            num_words = [len(sentence.split()) for sentence in sentences]
            text = copy.deepcopy(sentences)
            additional_tokens = []
            # If the tokenizer has a `cls_token`, we insert a `cls_token` at the beginning of the text
            # and a [SEP] token at the end of the text. For models without a `cls_token`, no tokens are inserted.
            use_special_tokens = self.tokenizer.cls_token is not None
            if use_special_tokens:
                additional_tokens += [self.tokenizer.cls_token, self.tokenizer.sep_token]
                if len(text) > 0:
                    text[0] = self.tokenizer.cls_token + text[0]
                    text[-1] = text[-1] + self.tokenizer.sep_token

            # Tokenized input
            tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in text]
            # chain
            tokenized_sentences = list(itertools.chain.from_iterable(tokenized_sentences))
            tokenized_sentences = np.array(tokenized_sentences)
            # mapping from original text to later undo chain
            sentence_indices = [0] + [sum(num_words[:i]) for i in range(1, len(num_words), 1)]

            # sliding window approach (see https://github.com/google-research/bert/issues/66)
            # however, since this is a brain model candidate, we don't let it see future words (just like the brain
            # doesn't receive future word input). Instead, we maximize the past context of each word
            sentence_index = 0
            sentences_chain = ' '.join(sentences).split()
            previous_indices = []

            encoded_layers = [[]] * len(self.layer_names)
            max_num_words = 512 if not use_special_tokens else 511
            for token_index in tqdm(range(len(tokenized_sentences)), desc='token features'):
                if tokenized_sentences[token_index] in additional_tokens:
                    continue  # ignore altogether
                # combine e.g. "'hunts', '##man'" or "'jennie', '##s'"
                tokens = [word.lstrip('##') for word in tokenized_sentences[previous_indices + [token_index]]]
                token_word = ''.join(tokens).lower()
                for special_token in self.tokenizer_special_tokens:
                    token_word = token_word.replace(special_token, '')
                if sentences_chain[sentence_index].lower() != token_word:
                    previous_indices.append(token_index)
                    continue
                previous_indices = []
                sentence_index += 1

                context_start = max(0, token_index - max_num_words + 1)
                context = tokenized_sentences[context_start:token_index + 1]
                if use_special_tokens and context_start > 0:  # `cls_token` has been discarded
                    # insert `cls_token` again following
                    # https://huggingface.co/pytorch-transformers/model_doc/roberta.html#pytorch_transformers.RobertaModel
                    context = np.insert(context, 0, tokenized_sentences[0])
                context_ids = self.tokenizer.convert_tokens_to_ids(context)

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([context_ids])

                # Predict hidden states features for each layer
                with torch.no_grad():
                    context_encoding, = self.model(tokens_tensor)[-1:]
                # We have a hidden state for all the layers
                assert len(context_encoding) == len(self.layer_names)
                # take only the encoding of the current word index
                word_encoding = [encoding[:, -1:, :] for encoding in context_encoding]
                word_encoding = [PytorchWrapper._tensor_to_numpy(encoding) for encoding in word_encoding]
                encoded_layers = [previous_words + [word_layer_encoding] for previous_words, word_layer_encoding
                                  in zip(encoded_layers, word_encoding)]
            encoded_layers = [np.concatenate(layer_encoding, axis=1) for layer_encoding in encoded_layers]
            assert all(layer_encoding.shape[1] == sum(num_words) for layer_encoding in encoded_layers)
            # separate into sentences again
            sentence_encodings = [[layer_encoding[:, start:end, :] for start, end in
                                   zip(sentence_indices, sentence_indices[1:] + [sum(num_words)])]
                                  for layer_encoding in encoded_layers]
            sentence_encodings = OrderedDict(zip(self.layer_names, sentence_encodings))
            sentence_encodings = OrderedDict([(layer, encoding) for layer, encoding in sentence_encodings.items()
                                              if layer in layers])
            return sentence_encodings


class KeyedVectorModel:
    """
    Lookup-table-like models where each word has an embedding.
    To retrieve the sentence activation, we take the mean of the word embeddings.
    """

    def __init__(self, identifier, weights_file, binary=False):
        super().__init__()
        from gensim.models.keyedvectors import KeyedVectors
        self._model = KeyedVectors.load_word2vec_format(weights_file, binary=binary)
        self._index2word_set = set(self._model.index2word)
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)
        self._extractor.register_activations_hook(word_mean)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['projection'])
        encoding = [np.array(self._encode_sentence(sentence)) for sentence in sentences]
        # expand "batch" dimension for compatibility with transformers (for sentence-word-aggregation)
        encoding = [np.expand_dims(sentence_encodings, 0) for sentence_encodings in encoding]
        return {'projection': encoding}

    def _encode_sentence(self, sentence):
        words = sentence.split()
        feature_vectors = []
        for word in words:
            if word in self._index2word_set:
                feature_vectors.append(self._model[word])
            else:
                self._logger.warning(f"Word {word} not present in model")
        return feature_vectors

    available_layers = ['projection']
    default_layers = available_layers


class Word2Vec(KeyedVectorModel):
    """
    https://arxiv.org/pdf/1310.4546.pdf
    """

    def __init__(self, weights_file='GoogleNews-vectors-negative300.bin'):
        weights_file = os.path.join(_ressources_dir, 'word2vec', weights_file)
        super(Word2Vec, self).__init__(identifier='word2vec', weights_file=weights_file, binary=True)


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
        super(Glove, self).__init__(identifier='glove', weights_file=word2vec_weightsfile)


#class RecursiveNeuralTensorNetwork(Model):
#    """
#    http://www.aclweb.org/anthology/D13-1170
#    """
#
#    def __init__(self, weights='sentiment'):
#        cachepath = os.path.join(_ressources_dir, 'recursive-neural-tensor-network', weights + '.activations.csv')
#        self._cache = pd.read_csv(cachepath)
#        self._cache = self._cache[self._cache['node.type'] == 'ROOT']
#        self._cache.drop_duplicates(inplace=True)
#
#    def __call__(self, sentences):
#        result = self._cache[self._cache['sentence'].isin(sentences)
#                             | self._cache['sentence'].isin([sentence + '.' for sentence in sentences])]
#        if len(result) != 1:
#            print(sentences)
#        assert len(result) == 1
#        result = result[[column for column in result if column.startswith('activation')]]
#        return result.values


def load_model(model_name):
    return model_pool[model_name]


model_pool = {
    'topicETM': LazyLoad(TopicETM),
    'random-gaussian': LazyLoad(GaussianRandom),
    'skip-thoughts': LazyLoad(SkipThoughts),
    'lm_1b': LazyLoad(LM1B),
    'word2vec': LazyLoad(Word2Vec),
    'glove': LazyLoad(Glove),
#    'rntn': LazyLoad(RecursiveNeuralTensorNetwork),
    'transformer-wordmean': LazyLoad(Transformer_WordMean),
    'transformer-wordall': LazyLoad(Transformer_WordAll),
    'transformer-wordlast': LazyLoad(Transformer_WordLast),
    'transformer-subsample_random': LazyLoad(Transformer_SubsampleRandom),
    'transformer-pad_zero': LazyLoad(Transformer_PadZero),
}
model_layers = {
    'topicETM': TopicETM.default_layers,
    'random-gaussian': GaussianRandom.default_layers,
    'skip-thoughts': SkipThoughts.default_layers,
    'lm_1b': LM1B.default_layers,
    'word2vec': Word2Vec.default_layers,
    'glove': Glove.default_layers,
    'transformer-wordmean': Transformer.default_layers,
    'transformer-wordall': Transformer.default_layers,
    'transformer-wordlast': Transformer.default_layers,
    'transformer-subsample_random': Transformer.default_layers,
    'transformer-pad_zero': Transformer.default_layers,
}

SPIECE_UNDERLINE = u'▁'  # define directly to avoid having to import (from pytorch_transformers.tokenization_xlnet)
transformers = [
    ('bert',
     'BertConfig', 'BertModel', 'BertTokenizer', (), 'bert-base-uncased',
     # https://github.com/huggingface/pytorch-pretrained-BERT/blob/78462aad6113d50063d8251e27dbaadb7f44fbf0/pytorch_pretrained_bert/modeling.py#L480
     ('embedding',) + tuple(f'encoder.layer.{i}.output' for i in range(12))  # output == layer_norm(fc(attn) + attn)
     ),
    ('bert-large',
     'BertConfig', 'BertModel', 'BertTokenizer', (), 'bert-large-uncased',
     # https://github.com/huggingface/pytorch-pretrained-BERT/blob/78462aad6113d50063d8251e27dbaadb7f44fbf0/pytorch_pretrained_bert/modeling.py#L480
     ('embedding',) + tuple(f'encoder.layer.{i}.output' for i in range(24))  # output == layer_norm(fc(attn) + attn)
     ),
    ('openaigpt',
     'OpenAIGPTConfig', 'OpenAIGPTModel', 'OpenAIGPTTokenizer', ('</w>',), 'openai-gpt',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_openai.py#L517
     ('drop',) + tuple(f'encoder.h.{i}.ln_2' for i in range(12))
     ),
    ('gpt2',
     'GPT2Config', 'GPT2Model', 'GPT2Tokenizer', ('ġ',), 'gpt2',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_gpt2.py#L514
     ('drop',) + tuple(f'encoder.h.{i}' for i in range(12))
     ),
    ('gpt2-medium',
     'GPT2Config', 'GPT2Model', 'GPT2Tokenizer', ('ġ',), 'gpt2-medium',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_gpt2.py#L514
     ('drop',) + tuple(f'encoder.h.{i}' for i in range(24))
     ),
    ('gpt2-large',
     'GPT2Config', 'GPT2Model', 'GPT2Tokenizer', ('ġ',), 'gpt2-large',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_gpt2.py#L514
     ('drop',) + tuple(f'encoder.h.{i}' for i in range(36))
     ),
    ('transfoxl',
     'TransfoXLConfig', 'TransfoXLModel', 'TransfoXLTokenizer', (), 'transfo-xl-wt103',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_transfo_xl.py#L1161
     ('drop',) + tuple(f'encoder.layers.{i}' for i in range(18))
     ),
    ('xlnet',
     'XLNetConfig', 'XLNetModel', 'XLNetTokenizer', (SPIECE_UNDERLINE,), 'xlnet-base-cased',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_xlnet.py#L962
     ('drop',) + tuple(f'encoder.layer.{i}' for i in range(12))
     ),
    ('xlnet-large',
     'XLNetConfig', 'XLNetModel', 'XLNetTokenizer', (SPIECE_UNDERLINE,), 'xlnet-large-cased',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_xlnet.py#L962
     ('drop',) + tuple(f'encoder.layer.{i}' for i in range(24))
     ),
    ('xlm',
     'XLMConfig', 'XLMModel', 'XLMTokenizer', ('</w>',), 'xlm-mlm-enfr-1024',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_xlm.py#L638
     ('dropout',) + tuple(f'encoder.layer_norm2.{i}' for i in range(6))
     ),
    ('xlm-clm',
     'XLMConfig', 'XLMModel', 'XLMTokenizer', ('</w>',), 'xlm-clm-enfr-1024',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_xlm.py#L638
     ('dropout',) + tuple(f'encoder.layer_norm2.{i}' for i in range(6))
     ),
    ('roberta',
     'RobertaConfig', 'RobertaModel', 'RobertaTokenizer', ('ġ',), 'roberta-base',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_roberta.py#L174
     ('embedding',) + tuple(f'encoder.layer.{i}' for i in range(12))
     ),
    ('roberta-large',
     'RobertaConfig', 'RobertaModel', 'RobertaTokenizer', ('ġ',), 'roberta-large',
     # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_roberta.py#L174
     ('embedding',) + tuple(f'encoder.layer.{i}' for i in range(24))
     ),
]
for untrained in False, True:
    for (identifier,
         config_ctr, model_ctr, tokenizer_ctr, tokenizer_special_tokens, pretrained_weights,
         layers) in transformers:

        if untrained:
            identifier += '-untrained'


        def ModelInstantiation(identifier=identifier, config_ctr=config_ctr, model_ctr=model_ctr,
                               tokenizer_ctr=tokenizer_ctr, tokenizer_special_tokens=tokenizer_special_tokens,
                               pretrained_weights=pretrained_weights, layers=layers,
                               untrained=untrained):
            module = import_module('pytorch_transformers')
            config_ctr = getattr(module, config_ctr)
            model_ctr, tokenizer_ctr = getattr(module, model_ctr), getattr(module, tokenizer_ctr)
            # Load pre-trained model tokenizer (vocabulary) and model
            config = config_ctr.from_pretrained(pretrained_weights)
            tokenizer = tokenizer_ctr.from_pretrained(pretrained_weights)
            state_dict = None
            if untrained:
                model = model_ctr(config=config)
                state_dict = model.state_dict()  # force loading of initial
            model = model_ctr.from_pretrained(pretrained_weights, output_hidden_states=True, state_dict=state_dict)
            transformer = _PytorchTransformerWrapper(identifier=identifier,
                                                     tokenizer=tokenizer,
                                                     tokenizer_special_tokens=tokenizer_special_tokens,
                                                     model=model, layers=layers)
            transformer._extractor.register_activations_hook(word_last)
            return transformer


        model_pool[identifier] = LazyLoad(ModelInstantiation)
        model_layers[identifier] = list(layers)
