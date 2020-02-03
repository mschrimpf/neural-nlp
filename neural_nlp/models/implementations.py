import copy
import os
import pickle
import tempfile
from collections import OrderedDict
from enum import Enum
from importlib import import_module

import itertools
import logging
import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from pathlib import Path
from tqdm import tqdm

from brainscore.utils import LazyLoad
from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
from neural_nlp.models.wrapper.pytorch import PytorchWrapper

_ressources_dir = (Path(__file__).parent / '..' / '..' / 'ressources' / 'models').resolve()

_logger = logging.getLogger(__name__)


class BrainModel:
    Modes = Enum('Mode', 'recording tokens_to_features')

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

    def tokenize(self, text):
        raise NotImplementedError()

    def tokens_to_inputs(self, tokens):
        return tokens

    @property
    def features_size(self):
        raise NotImplementedError()

    @property
    def vocab_size(self):
        raise NotImplementedError()


class SentenceLength(BrainModel):
    available_layers = ['sentence-length']
    default_layers = available_layers

    identifier = 'sentence-length'

    def __init__(self):
        super(SentenceLength, self).__init__()
        self._extractor = ActivationsExtractorHelper(identifier=self.identifier,
                                                     get_activations=self._get_activations, reset=lambda: None)

    def __call__(self, *args, average_sentence=True, **kwargs):
        if not average_sentence:
            raise ValueError("This model only works on a sentence-level")
        return self._extractor(*args, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, self.available_layers)
        sentence_lengths = [len(sentence.split(' ')) for sentence in sentences]
        return {self.available_layers[0]: np.array(sentence_lengths)}


class TopicETM:
    """https://arxiv.org/abs/1907.04907"""

    def __init__(self):
        super().__init__()
        weights_file = os.path.join(_ressources_dir, 'topicETM', 'normalized_betas_50K.npy')
        vocab_file = os.path.join(_ressources_dir, 'topicETM', 'vocab_50K.pkl')
        self.weights = np.load(weights_file)
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)

        wordEmb_TopicSpace = {}
        for elm in tqdm(self.vocab, desc='vocab'):
            i = self.vocab.index(elm)  # get index of word
            wordEmb_TopicSpace[elm] = self.weights[:, i]
        self.wordEmb_TopicSpace = wordEmb_TopicSpace
        self._extractor = ActivationsExtractorHelper(identifier='topicETM', get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, average_sentence=True, **kwargs):
        return _call_conditional_average(*args, extractor=self._extractor,
                                         average_sentence=average_sentence, sentence_averaging=word_mean, **kwargs)

    def _encode_sentence(self, sentence):
        words = sentence.split()
        feature_vectors = []
        for word in words:
            if word in self.vocab:
                feature_vectors.append(self.wordEmb_TopicSpace[word])
            else:
                self._logger.warning(f"Word {word} not present in model")
                feature_vectors.append(np.zeros((100,)))
        return feature_vectors

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['projection'])
        encoding = [np.array(self._encode_sentence(sentence)) for sentence in sentences]
        encoding = [np.expand_dims(sentence_encodings, 0) for sentence_encodings in encoding]
        return {'projection': encoding}

    available_layers = ['projection']
    default_layers = available_layers


class SkipThoughts:
    """
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """

    identifier = 'skip-thoughts'

    def __init__(self, weights=os.path.join(_ressources_dir, 'skip-thoughts')):
        super().__init__()
        import skipthoughts
        weights = weights + '/'
        model = LazyLoad(lambda: skipthoughts.load_model(path_to_models=weights, path_to_tables=weights))
        self._encoder = LazyLoad(lambda: skipthoughts.Encoder(model))
        self._extractor = ActivationsExtractorHelper(identifier=self.identifier, get_activations=self._get_activations,
                                                     reset=lambda: None)  # not sure if this resets states on its own
        self._extractor.insert_attrs(self)

    def __call__(self, *args, average_sentence=True, **kwargs):
        return _call_conditional_average(*args, extractor=self._extractor,
                                         average_sentence=average_sentence, sentence_averaging=word_last, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['encoder'])
        encoding = []
        for sentence_iter, sentence in enumerate(sentences):
            sentence_words = []
            encoding.append([])
            for word in sentence.split(' '):
                sentence_words.append(word)
                word_embeddings = self._encoder.encode([' '.join(sentence_words)])
                encoding[sentence_iter].append(word_embeddings)
            encoding[sentence_iter] = np.array(encoding[sentence_iter]).transpose([1, 0, 2])
        return {'encoder': encoding}

    available_layers = ['encoder']
    default_layers = available_layers


class LM1B:
    """
    https://arxiv.org/pdf/1602.02410.pdf
    """

    identifier = 'lm_1b'

    def __init__(self, weights=os.path.join(_ressources_dir, 'lm_1b')):
        super().__init__()
        from lm_1b.lm_1b_eval import Encoder
        self._encoder = Encoder(vocab_file=os.path.join(weights, 'vocab-2016-09-10.txt'),
                                pbtxt=os.path.join(weights, 'graph-2016-09-10.pbtxt'),
                                ckpt=os.path.join(weights, 'ckpt-*'))
        self._extractor = ActivationsExtractorHelper(identifier=self.identifier, get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, average_sentence=True, **kwargs):
        return _call_conditional_average(*args, extractor=self._extractor,
                                         average_sentence=average_sentence, sentence_averaging=word_last, **kwargs)

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
                if i > 0:  # 0 is <S>
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

    def __call__(self, *args, average_sentence=True, **kwargs):
        return _call_conditional_average(*args, extractor=self._extractor,
                                         average_sentence=average_sentence, sentence_averaging=word_last, **kwargs)

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
    def __init__(self, identifier, tokenizer, model, layers, sentence_average, tokenizer_special_tokens=()):
        super(_PytorchTransformerWrapper, self).__init__()
        self.default_layers = self.available_layers = layers
        self._tokenizer = tokenizer
        self._model = model
        self._model_container = self.ModelContainer(tokenizer, model, layers, tokenizer_special_tokens)
        self._sentence_average = sentence_average
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self._model_container,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)

    def __call__(self, *args, average_sentence=True, **kwargs):
        self._model.eval()
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(*args, extractor=self._extractor, average_sentence=average_sentence,
                                             sentence_averaging=self._sentence_average, **kwargs)
        elif self.mode == BrainModel.Modes.tokens_to_features:
            return self._tokens_to_features(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def _tokens_to_features(self, token_ids):
        import torch
        max_num_words = 512 - 2  # -2 for [cls], [sep]
        features = []
        for token_index in range(len(token_ids)):
            context_start = max(0, token_index - max_num_words + 1)
            context_ids = token_ids[context_start:token_index + 1]
            tokens_tensor = torch.tensor([context_ids])
            tokens_tensor = tokens_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
            context_features = self._model(tokens_tensor)[0]
            features.append(PytorchWrapper._tensor_to_numpy(context_features[:, -1, :]))
        return np.concatenate(features)

    @property
    def identifier(self):
        return self._extractor.identifier

    def tokenize(self, text, vocab_size=None):
        assert not (bool(vocab_size)) or vocab_size == self.vocab_size
        tokenized_text = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
        tokenized_text = np.array(tokenized_text)  # ~10 sec with numpy, ~40 hours without
        return tokenized_text

    def tokens_to_inputs(self, tokens):
        return self._tokenizer.build_inputs_with_special_tokens(tokens)

    @property
    def features_size(self):
        return self._model.config.hidden_size

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    class ModelContainer:
        def __init__(self, tokenizer, model, layer_names, tokenizer_special_tokens=()):
            import torch
            self.tokenizer = tokenizer
            self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
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

            max_num_words = 512 if not use_special_tokens else 511
            aligned_tokens = self.align_tokens(
                tokenized_sentences=tokenized_sentences, sentences=sentences,
                max_num_words=max_num_words, additional_tokens=additional_tokens, use_special_tokens=use_special_tokens)
            encoded_layers = [[]] * len(self.layer_names)
            for context_ids in aligned_tokens:
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([context_ids])
                tokens_tensor = tokens_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

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

        def align_tokens(self, tokenized_sentences, sentences, max_num_words, additional_tokens, use_special_tokens):
            # sliding window approach (see https://github.com/google-research/bert/issues/66)
            # however, since this is a brain model candidate, we don't let it see future words (just like the brain
            # doesn't receive future word input). Instead, we maximize the past context of each word
            sentence_index = 0
            sentences_chain = ' '.join(sentences).split()
            previous_indices = []

            for token_index in tqdm(range(len(tokenized_sentences)), desc='token features'):
                if tokenized_sentences[token_index] in additional_tokens:
                    continue  # ignore altogether
                # combine e.g. "'hunts', '##man'" or "'jennie', '##s'"
                tokens = [word.lstrip('##').lstrip('▁')  # tokens are sometimes padded by prefixes, clear those again
                          for word in tokenized_sentences[previous_indices + [token_index]]]
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
                yield context_ids


class KeyedVectorModel(BrainModel):
    """
    Lookup-table-like models where each word has an embedding.
    To retrieve the sentence activation, we take the mean of the word embeddings.
    """

    available_layers = ['projection']
    default_layers = available_layers

    def __init__(self, identifier, weights_file, random_embeddings=False, random_std=1, binary=False):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        from gensim.models.keyedvectors import KeyedVectors
        self._model = KeyedVectors.load_word2vec_format(weights_file, binary=binary)
        self._index2word_set = set(self._model.index2word)
        if random_embeddings:
            self._logger.debug(f"Replacing embeddings with random N(0, {random_std})")
            random_embedding = RandomState(0).randn(len(self._index2word_set), len(self._model['the'])) * random_std
            self._model = {word: random_embedding[i] for i, word in enumerate(sorted(self._index2word_set))}
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)

    def __call__(self, sentences, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(sentences, *args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_mean, **kwargs)
        elif self.mode == BrainModel.Modes.tokens_to_features:
            sentences = " ".join(self._model.index2word[index] for index in sentences)
            return self._encode_sentence(sentences, *args, **kwargs)

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
                feature_vectors.append(np.zeros((300,)))
        return feature_vectors

    def tokenize(self, text, vocab_size=None):
        vocab_size = vocab_size or self.vocab_size
        tokens = [self._model.vocab[word].index for word in text.split() if word in self._model.vocab
                  and self._model.vocab[word].index < vocab_size]  # only top-k vocab words
        return np.array(tokens)

    @property
    def vocab_size(self):
        return len(self._model.vocab)

    @property
    def features_size(self):
        return 300


class Word2Vec(KeyedVectorModel):
    """
    https://arxiv.org/pdf/1310.4546.pdf
    """

    identifier = 'word2vec'

    def __init__(self, weights_file='GoogleNews-vectors-negative300.bin', **kwargs):
        weights_file = os.path.join(_ressources_dir, 'word2vec', weights_file)
        super(Word2Vec, self).__init__(
            identifier=self.identifier, weights_file=weights_file, binary=True,
            # standard embedding std
            # https://github.com/pytorch/pytorch/blob/ecbf6f99e6a4e373105133b31534c9fb50f2acca/torch/nn/modules/sparse.py#L106
            random_std=1, **kwargs)


class Glove(KeyedVectorModel):
    """
    http://www.aclweb.org/anthology/D14-1162
    """

    identifier = 'glove'

    def __init__(self, weights='glove.840B.300d.txt', **kwargs):
        from gensim.scripts.glove2word2vec import glove2word2vec
        weights_file = os.path.join(_ressources_dir, 'glove', weights)
        word2vec_weightsfile = weights_file + '.word2vec'
        if not os.path.isfile(word2vec_weightsfile):
            glove2word2vec(weights_file, word2vec_weightsfile)
        super(Glove, self).__init__(
            identifier=self.identifier, weights_file=word2vec_weightsfile,
            # std from https://gist.github.com/MatthieuBizien/de26a7a2663f00ca16d8d2558815e9a6#file-fast_glove-py-L16
            random_std=.01, **kwargs)


class RecursiveNeuralTensorNetwork(BrainModel):
    """
    http://www.aclweb.org/anthology/D13-1170
    """

    def __init__(self, weights='sentiment'):
        cachepath = os.path.join(_ressources_dir, 'recursive-neural-tensor-network', weights + '.activations.csv')
        self._cache = pd.read_csv(cachepath)
        self._cache = self._cache[self._cache['node.type'] == 'ROOT']
        self._cache.drop_duplicates(inplace=True)

    def __call__(self, sentences):
        result = self._cache[self._cache['sentence'].isin(sentences)
                             | self._cache['sentence'].isin([sentence + '.' for sentence in sentences])]
        if len(result) != 1:
            print(sentences)
        assert len(result) == 1
        result = result[[column for column in result if column.startswith('activation')]]
        return result.values


def _call_conditional_average(*args, extractor, average_sentence, sentence_averaging, **kwargs):
    if average_sentence:
        handle = extractor.register_activations_hook(sentence_averaging)
    result = extractor(*args, **kwargs)
    if average_sentence:
        handle.remove()
    return result


def load_model(model_name):
    return model_pool[model_name]


model_pool = {
    SentenceLength.identifier: LazyLoad(SentenceLength),
    SkipThoughts.identifier: LazyLoad(SkipThoughts),
    LM1B.identifier: LazyLoad(LM1B),
    Word2Vec.identifier: LazyLoad(Word2Vec),
    Word2Vec.identifier + '-untrained': LazyLoad(lambda: Word2Vec(random_embeddings=True)),
    Glove.identifier: LazyLoad(Glove),
    Glove.identifier + '-untrained': LazyLoad(lambda: Glove(random_embeddings=True)),
    'transformer': LazyLoad(Transformer),
    'topicETM': LazyLoad(TopicETM),
}
model_layers = {
    SentenceLength.identifier: SentenceLength.default_layers,
    SkipThoughts.identifier: SkipThoughts.default_layers,
    LM1B.identifier: LM1B.default_layers,
    Word2Vec.identifier: Word2Vec.default_layers,
    Glove.identifier: Glove.default_layers,
    'transformer': Transformer.default_layers,
    'topicETM': TopicETM.default_layers,
}
# untrained layers are the same as trained ones
model_layers = {**model_layers, **{f"{identifier}-untrained": layers for identifier, layers in model_layers.items()}}

SPIECE_UNDERLINE = u'▁'  # define directly to avoid having to import (from pytorch_transformers.tokenization_xlnet)
transformer_configurations = []
"""
Each model configuration is a dictionary with the following keys:
- identifier: if empty, use weight_identifier)
- weight_identifier: which pretrained weights to use
- prefix: the model's string prefix from which to build <prefix>Config, <prefix>Model, and <prefix>Tokenizer 
    if they are not defined.
- config_ctr: the importable class name of the model's config class
- model_ctr: the importable class name of the model's model class
- tokenizer_ctr: the importable class name of the model's tokenizer class
- layers: a list of layers from which we will retrieve activations
"""
# bert
for identifier, num_layers in [
    ('bert-base-uncased', 12),
    ('bert-base-multilingual-cased', 12),
    ('bert-large-uncased', 24),
    ('bert-large-uncased-whole-word-masking', 24),
]:
    transformer_configurations.append(dict(
        prefix='Bert', weight_identifier=identifier,
        # https://github.com/huggingface/pytorch-pretrained-BERT/blob/78462aad6113d50063d8251e27dbaadb7f44fbf0/pytorch_pretrained_bert/modeling.py#L480
        # output == layer_norm(fc(attn) + attn)
        layers=('embedding',) + tuple(f'encoder.layer.{i}.output' for i in range(num_layers))
    ))
# openaigpt
transformer_configurations.append(dict(
    prefix='OpenAIGPT', identifier='openaigpt', weight_identifier='openai-gpt', tokenizer_special_tokens=('</w>',),
    # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_openai.py#L517
    layers=('drop',) + tuple(f'encoder.h.{i}.ln_2' for i in range(12))
))
# gpt2
for identifier, num_layers in [
    ('gpt2', 12),
    ('gpt2-medium', 24),
    ('gpt2-large', 36),
    ('gpt2-xl', 48),
    ('distilgpt2', 6),
]:
    transformer_configurations.append(dict(
        prefix='GPT2', weight_identifier=identifier, tokenizer_special_tokens=('ġ',),
        # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_gpt2.py#L514
        layers=('drop',) + tuple(f'encoder.h.{i}' for i in range(num_layers))
    ))
# transformer xl
transformer_configurations.append(dict(
    prefix='TransfoXL', weight_identifier='transfo-xl-wt103',
    # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_transfo_xl.py#L1161
    layers=('drop',) + tuple(f'encoder.layers.{i}' for i in range(18))
))
# xlnet
for identifier, num_layers in [
    ('xlnet-base-cased', 12),
    ('xlnet-large-cased', 24),
]:
    transformer_configurations.append(dict(
        prefix='XLNet', tokenizer_special_tokens=(SPIECE_UNDERLINE,), weight_identifier=identifier,
        # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_xlnet.py#L962
        layers=('drop',) + tuple(f'encoder.layer.{i}' for i in range(num_layers))
    ))
# xlm
for identifier, num_layers in [
    ('xlm-mlm-en-2048', 12),
    ('xlm-mlm-enfr-1024', 6),
    ('xlm-mlm-xnli15-1024', 12),
    ('xlm-clm-enfr-1024', 6),
    ('xlm-mlm-100-1280', 16),
]:
    transformer_configurations.append(dict(
        prefix='XLM', tokenizer_special_tokens=('</w>',), weight_identifier=identifier,
        # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_xlm.py#L638
        layers=('dropout',) + tuple(f'encoder.layer_norm2.{i}' for i in range(num_layers))
    ))
# roberta
for identifier, num_layers in [
    ('roberta-base', 12),
    ('roberta-large', 24),
    ('distilroberta-base', 6),
]:
    transformer_configurations.append(dict(
        prefix='Roberta', tokenizer_special_tokens=('ġ',), weight_identifier=identifier,
        # https://github.com/huggingface/pytorch-transformers/blob/c589862b783b94a8408b40c6dc9bf4a14b2ee391/pytorch_transformers/modeling_roberta.py#L174
        layers=('embedding',) + tuple(f'encoder.layer.{i}' for i in range(num_layers))
    ))
# distilbert
for identifier, num_layers in [
    ('distilbert-base-uncased', 6),
]:
    transformer_configurations.append(dict(
        prefix='DistilBert', tokenizer_special_tokens=('ġ',), weight_identifier=identifier,
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_distilbert.py#L482
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_distilbert.py#L258
        layers=('embeddings',) + tuple(f'transformer.layer.{i}' for i in range(num_layers))
    ))
# ctrl
transformer_configurations.append(dict(
    prefix='CTRL', tokenizer_special_tokens=('ġ',), weight_identifier='ctrl',
    # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_ctrl.py#L388
    # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_ctrl.py#L408
    layers=('w',) + tuple(f'h.{i}' for i in range(48))
))
# albert
for (identifier, num_layers), version in itertools.product([
    ('albert-base', 12),
    ('albert-large', 24),
    ('albert-xlarge', 24),
    ('albert-xxlarge', 12),
], [1, 2]):
    identifier = f"{identifier}-v{version}"
    transformer_configurations.append(dict(
        prefix='Albert', tokenizer_special_tokens=('ġ',), weight_identifier=identifier,
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_albert.py#L557
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_albert.py#L335
        layers=('embeddings',) + tuple(f'encoder.albert_layer_groups.{i}' for i in range(num_layers))
    ))


# t5
class _T5Wrapper:
    def __init__(self, model):
        self._model = model

    def __call__(self, tokens_tensor):
        # the decoder_input_ids are not right, but we only retrieve encoder features anyway
        return self._model(encoder_input_ids=tokens_tensor, decoder_input_ids=tokens_tensor)

    def __getattr__(self, item):  # forward attribute retrieval
        if item == '_model':
            return super(_T5Wrapper, self).__getattr__(item)
        return getattr(self._model, item)


for identifier, num_layers in [
    ('t5-small', 6),
    ('t5-base', 12),
    ('t5-large', 24),
    ('t5-3b', 24),
    ('t5-11b', 24),
]:
    transformer_configurations.append(dict(
        prefix='T5', tokenizer_special_tokens=('ġ',), weight_identifier=identifier,
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_t5.py#L773
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_t5.py#L605
        layers=('shared',) + tuple(f'encoder.block.{i}' for i in range(num_layers)),
        model_wrapper=_T5Wrapper,
    ))
# xlm-roberta
for identifier, num_layers in [
    ('xlm-roberta-base', 12),
    ('xlm-roberta-large', 24),
]:
    transformer_configurations.append(dict(
        prefix='XLMRoberta', tokenizer_special_tokens=('ġ',), weight_identifier=identifier,
        # https://github.com/huggingface/transformers/blob/80faf22b4ac194061a08fde09ad8b202118c151e/src/transformers/modeling_xlm_roberta.py#L119
        layers=('embedding',) + tuple(f'encoder.layer.{i}' for i in range(num_layers))
    ))

for untrained in False, True:
    for configuration in transformer_configurations:
        configuration = copy.deepcopy(configuration)
        # either use the defined identifier or the weights used
        identifier = configuration.get('identifier', configuration['weight_identifier'])

        if untrained:
            identifier += '-untrained'
            configuration['trained'] = False

        # either use the defined values for config, model and tokenizer or build from prefix
        configuration['config_ctr'] = configuration.get('config_ctr', configuration['prefix'] + 'Config')
        configuration['model_ctr'] = configuration.get('model_ctr', configuration['prefix'] + 'Model')
        configuration['tokenizer_ctr'] = configuration.get('tokenizer_ctr', configuration['prefix'] + 'Tokenizer')


        def model_instantiation(identifier=identifier, configuration=frozenset(configuration.items())):
            configuration = dict(configuration)  # restore from frozen
            module = import_module('transformers')
            config_ctr = getattr(module, configuration['config_ctr'])
            model_ctr = getattr(module, configuration['model_ctr'])
            tokenizer_ctr = getattr(module, configuration['tokenizer_ctr'])
            # Load pre-trained model tokenizer (vocabulary) and model
            config = config_ctr.from_pretrained(configuration['weight_identifier'])
            tokenizer = tokenizer_ctr.from_pretrained(configuration['weight_identifier'])
            state_dict = None
            if not configuration.get('trained', True):  # if untrained
                model = model_ctr(config=config)
                state_dict = model.state_dict()  # force loading of initial
            model = model_ctr.from_pretrained(configuration['weight_identifier'],
                                              output_hidden_states=True, state_dict=state_dict)
            model_wrapper = configuration.get('model_wrapper', None)
            if model_wrapper:
                model = model_wrapper(model)
            transformer = _PytorchTransformerWrapper(
                identifier=identifier,
                tokenizer=tokenizer, tokenizer_special_tokens=configuration.get('tokenizer_special_tokens', ()),
                model=model, layers=configuration['layers'],
                sentence_average=word_last)
            return transformer


        model_pool[identifier] = LazyLoad(model_instantiation)
        model_layers[identifier] = list(configuration['layers'])
