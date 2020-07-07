import copy
from collections import OrderedDict, defaultdict
from enum import Enum
from importlib import import_module

import itertools
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tempfile
from brainio_collection.fetch import fullname
from numpy.random.mtrand import RandomState
from pathlib import Path
from tqdm import tqdm

from brainscore.utils import LazyLoad
from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
from neural_nlp.models.wrapper.pytorch import PytorchWrapper

_ressources_dir = (Path(__file__).parent / '..' / '..' / 'ressources' / 'models').resolve()


class BrainModel:
    Modes = Enum('Mode', 'recording')

    def __call__(self, sentences):
        """
        Record representations in response to sentences. Ideally this would be localized to a
        :param sentences:
        :return:
        """
        raise NotImplementedError()


class TaskModel:
    Modes = Enum('Mode', 'tokens_to_features sentence_features')

    def __init__(self):
        super(TaskModel, self).__init__()
        self._mode = BrainModel.Modes.recording  # run as BrainModel by default

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in TaskModel.Modes or value in BrainModel.Modes
        self._mode = value

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

    def glue_dataset(self, task, examples, label_list, output_mode, max_seq_length):
        """
        :return: a torch TensorDataset where the last item is the labels
        """
        raise NotImplementedError()


class SentenceLength(BrainModel, TaskModel):
    """
    control model
    """
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


class WordPosition(BrainModel):
    """
    control model
    """
    available_layers = ['word-position']
    default_layers = available_layers

    identifier = 'word-position'

    def __init__(self):
        super(WordPosition, self).__init__()
        self._extractor = ActivationsExtractorHelper(identifier=self.identifier,
                                                     get_activations=self._get_activations, reset=lambda: None)

    def __call__(self, *args, average_sentence=True, **kwargs):
        if average_sentence:
            raise ValueError("This model only works on a word-level")
        return self._extractor(*args, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, self.available_layers)
        word_positions = [np.array([[[i] for i, word in enumerate(sentence.split(' '))]]) for sentence in sentences]
        return {self.available_layers[0]: word_positions}


class RandomEmbedding(BrainModel):
    """
    control model
    """
    identifier = 'random-embedding'
    available_layers = [identifier]
    default_layers = available_layers

    def __init__(self, num_embeddings=1600):
        self._random_state = RandomState(0)
        self._embeddings = defaultdict(lambda: self._random_state.rand(num_embeddings))
        self._extractor = ActivationsExtractorHelper(identifier=self.identifier,
                                                     get_activations=self._get_activations, reset=lambda: None)

    def __call__(self, *args, average_sentence=True, **kwargs):
        return _call_conditional_average(*args, extractor=self._extractor,
                                         average_sentence=average_sentence, sentence_averaging=word_mean, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, self.available_layers)
        word_embeddings = [np.array([[self._embeddings[word] for word in sentence.split()]]) for sentence in sentences]
        return {self.available_layers[0]: word_embeddings}


class ETM(BrainModel, TaskModel):
    """
    Dieng et al., 2019
    https://arxiv.org/abs/1907.04907
    """

    identifier = 'ETM'

    available_layers = ['projection']
    default_layers = available_layers

    def __init__(self, weights_file='rho_100_20ng_min_df_2.npy', vocab_file='vocab_100_20ng_min_df_2.pkl',
                 emb_size=300, random_embeddings=False, random_std=1):
        super().__init__()
        self._logger = logging.getLogger(fullname(self))

        weights_file = os.path.join(_ressources_dir, 'topicETM', weights_file)
        vocab_file = os.path.join(_ressources_dir, 'topicETM', vocab_file)
        self.weights = np.load(weights_file)
        self.emb_size = emb_size
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        self.vocab_index = {word: index for index, word in enumerate(self.vocab)}
        self.index_vocab = {index: word for index, word in enumerate(self.vocab)}

        if random_embeddings:
            self._logger.debug(f"Replacing embeddings with random N(0, {random_std})")
            random_embedding = RandomState(0).randn(len(self.vocab), self.emb_size) * random_std
            self.wordEmb_TopicSpace = {word: random_embedding[i] for i, word in enumerate(sorted(self.vocab))}
        else:
            wordEmb_TopicSpace = {}
            for elm in tqdm(self.vocab, desc='vocab'):
                i = self.vocab.index(elm)  # get index of word
                wordEmb_TopicSpace[elm] = self.weights[i]
            self.wordEmb_TopicSpace = wordEmb_TopicSpace

        self._extractor = ActivationsExtractorHelper(
            identifier=self.identifier + ('-untrained' if random_embeddings else ''),
            get_activations=self._get_activations, reset=lambda: None)
        self._extractor.insert_attrs(self)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __call__(self, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(*args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_mean, **kwargs)
        elif self.mode == TaskModel.Modes.tokens_to_features:
            return self._encode_sentence(*args, **kwargs)

    def _encode_sentence(self, sentence):
        if isinstance(sentence, str):
            words = sentence.split()
        else:
            words = [self.index_vocab[index] for index in sentence]
        feature_vectors = []
        for word in words:
            word = word.lower()
            if word in self.vocab:
                feature_vectors.append(self.wordEmb_TopicSpace[word])
            else:
                self._logger.warning(f"Word {word} not present in model")
                feature_vectors.append(np.zeros((self.emb_size,)))
        return feature_vectors

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, ['projection'])
        encoding = [np.array(self._encode_sentence(sentence)) for sentence in sentences]
        encoding = [np.expand_dims(sentence_encodings, 0) for sentence_encodings in encoding]
        return {'projection': encoding}

    def tokenize(self, text, vocab_size=None):
        vocab_size = vocab_size or self.vocab_size
        words = text.split()
        tokens = [self.vocab_index[word] for word in tqdm(words, desc='tokenize') if word in self.vocab
                  and self.vocab_index[word] < vocab_size]  # only top-k vocab words
        return np.array(tokens)

    @property
    def features_size(self):
        return self.emb_size

    @property
    def vocab_size(self):
        return len(self.vocab)


class SkipThoughts(BrainModel, TaskModel):
    """
    Kiros et al., 2015
    http://papers.nips.cc/paper/5950-skip-thought-vectors
    """

    identifier = 'skip-thoughts'

    def __init__(self, weights=os.path.join(_ressources_dir, 'skip-thoughts'), load_weights=True):
        super().__init__()
        self._logger = logging.getLogger(fullname(self))
        import skipthoughts
        weights = weights + '/'
        model = LazyLoad(lambda: skipthoughts.load_model(path_to_models=weights, path_to_tables=weights))
        self._encoder = LazyLoad(lambda: skipthoughts.Encoder(model))
        self._extractor = ActivationsExtractorHelper(
            identifier=self.identifier + ("-untrained" if not load_weights else ''),
            get_activations=self._get_activations, reset=lambda: None)  # resets states on its own
        self._extractor.insert_attrs(self)
        # setup prioritized vocabulary entries to map to indices and back.
        # unfortunately it does not seem straight-forward to retrieve preferred words/tokens for the model, such as
        # their frequency in the training dataset. The pretrained version we're using was trained on the BookCorpus
        # dataset which is no longer online and any word frequencies did not seem to be saved in the files available
        # here. The `encoder._model['utable']` is an OrderedDict, but the ordering does not intuitively seem to
        # correspond to frequencies. Since there seems to be no other way, we will treat the utable as a
        # frequency-ordered dict regardless and hope for the best.
        self.vocab_index = {word: index for index, word in enumerate(self._encoder._model['utable'])}
        self.index_vocab = {index: word for index, word in enumerate(self._encoder._model['utable'])}

    @property
    def vocab_size(self):
        return len(self._encoder._model['utable'])

    @property
    def features_size(self):
        return 4800

    def tokenize(self, text, vocab_size=None):
        if (bool(vocab_size)) and vocab_size < self.vocab_size:  # smaller vocab size requested, drop tokens
            self._logger.debug(f"Shortening {self.vocab_size} to {vocab_size}")
            _vocab_size = vocab_size
        else:
            _vocab_size = self.vocab_size
        words = text.split()
        tokens = [self.vocab_index[word] for word in tqdm(words, desc='tokenize') if word in self.vocab_index
                  and self.vocab_index[word] < _vocab_size]  # only top-k vocab words
        return np.array(tokens)

    def __call__(self, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(*args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_last, **kwargs)
        elif self.mode == TaskModel.Modes.tokens_to_features:
            return self._encode_sentence(*args, **kwargs)

    def _get_activations(self, sentences, layers):
        np.testing.assert_array_equal(layers, self.available_layers)
        encoding = [self._encode_sentence(sentence) for sentence in sentences]
        return {'encoder': encoding}

    def _encode_sentence(self, sentence):
        if isinstance(sentence, str):
            words = sentence.split(' ')
        else:
            words = [self.index_vocab[index] for index in sentence]
        sentence_words = []
        sentence_encoding = []
        for word in words:
            sentence_words.append(word)
            word_embeddings = self._encoder.encode([' '.join(sentence_words)])
            sentence_encoding.append(word_embeddings)
        sentence_encoding = np.array(sentence_encoding).transpose([1, 0, 2])
        return sentence_encoding

    available_layers = ['encoder']
    default_layers = available_layers


class LM1B(BrainModel, TaskModel):
    """
    Jozefowicz et al., 2016
    https://arxiv.org/pdf/1602.02410.pdf
    """

    identifier = 'lm_1b'

    def __init__(self, weights=os.path.join(_ressources_dir, 'lm_1b'), reset_weights=False):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        from lm_1b.lm_1b_eval import Encoder
        self._encoder = Encoder(vocab_file=os.path.join(weights, 'vocab-2016-09-10.txt'),
                                pbtxt=os.path.join(weights, 'graph-2016-09-10.pbtxt'),
                                ckpt=os.path.join(weights, 'ckpt-*'),
                                reset_weights=reset_weights)
        self._extractor = ActivationsExtractorHelper(
            identifier=self.identifier + ('-untrained' if reset_weights else ''),
            get_activations=self._get_activations, reset=self._initialize)
        self._extractor.insert_attrs(self)
        self._vocab_index = self._encoder.vocab._word_to_id
        self._index_vocab = {index: word for word, index in self._vocab_index.items()}

    @property
    def vocab_size(self):
        return len(self._vocab_index)

    @property
    def features_size(self):
        return 1024

    def tokenize(self, text, vocab_size=None):
        if (bool(vocab_size)) and vocab_size < self.vocab_size:  # smaller vocab size requested, drop tokens
            self._logger.debug(f"Shortening {self.vocab_size} to {vocab_size}")
            _vocab_size = vocab_size
        else:
            _vocab_size = self.vocab_size
        words = text.split()
        tokens = [self._vocab_index[word] for word in tqdm(words, desc='tokenize')
                  if word in self._vocab_index and self._vocab_index[word] < _vocab_size]  # only top-k vocab words
        return np.array(tokens)

    def __call__(self, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(*args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_last, **kwargs)
        elif self.mode == TaskModel.Modes.tokens_to_features:
            self._initialize()  # reset
            readout_layer = self.default_layers[-1]
            return self._encode_sentence(*args, layers=[readout_layer], **kwargs)[readout_layer]

    def _get_activations(self, sentences, layers):
        layer_activations = defaultdict(list)
        for sentence in sentences:
            embeddings = self._encode_sentence(sentence, layers)
            for layer, layer_embeddings in embeddings.items():
                layer_activations[layer].append(np.array([layer_embeddings]))
        return layer_activations

    def _encode_sentence(self, sentence, layers):
        from lm_1b import lm_1b_eval
        from six.moves import xrange
        if not isinstance(sentence, str):
            sentence = ' '.join([self._index_vocab[index] for index in sentence])
        self._initialize()
        # the following is copied from lm_1b.lm_1b_eval.Encoder.__call__.
        # only the `sess.run` call needs to be changed but there's no way to access it outside the code
        if sentence.find('<S>') != 0:
            sentence = '<S> ' + sentence
        word_ids = [self._encoder.vocab.word_to_id(w) for w in sentence.split()]
        char_ids = [self._encoder.vocab.word_to_char_ids(w) for w in sentence.split()]
        # some unknown characters end up as '�' (ord 65533). Replace those with empty (4)
        char_ids = [np.array([c if c < 256 else 4 for c in chars]) for chars in char_ids]
        inputs = np.zeros([lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros(
            [lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS, self._encoder.vocab.max_word_length], np.int32)
        embeddings = []
        targets = np.zeros([lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS], np.int32)
        weights = np.ones([lm_1b_eval.BATCH_SIZE, lm_1b_eval.NUM_TIMESTEPS], np.float32)
        for i in xrange(len(word_ids)):
            inputs[0, 0] = word_ids[i]
            char_ids_inputs[0, 0, :] = char_ids[i]
            # calling this repeatedly with the same input leads to different embeddings,
            # so we infer this preserves hidden state
            lstm_emb = self._encoder.sess.run([self._encoder.t[name] for name in layers],
                                              feed_dict={self._encoder.t['char_inputs_in']: char_ids_inputs,
                                                         self._encoder.t['inputs_in']: inputs,
                                                         self._encoder.t['targets_in']: targets,
                                                         self._encoder.t['target_weights_in']: weights})
            if i > 0:  # 0 is <S>
                embeddings.append(lstm_emb)
        # `embeddings` shape is now: words x layers x *layer_shapes
        layer_activations = {}
        for i, layer in enumerate(layers):
            # embeddings is `words x layers x (1 x 1024)`
            layer_activations[layer] = [embedding[i] for embedding in embeddings]
            # words x 1 x 1024 --> words x 1024
            layer_activations[layer] = np.array(layer_activations[layer]).transpose(1, 0, 2).squeeze(axis=0)
        return layer_activations

    def _initialize(self):
        self._encoder.sess.run(self._encoder.t['states_init'])

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


class Transformer(PytorchWrapper, BrainModel, TaskModel):
    """
    Vaswani & Shazeer & Parmar & Uszkoreit & Jones & Gomez & Kaiser & Polosukhin, 2017
    https://arxiv.org/pdf/1706.03762.pdf
    """

    identifier = 'transformer'

    def __init__(self, untrained=False):
        weights = os.path.join(_ressources_dir, 'transformer/averaged-10-epoch.pt')
        from onmt.opts import add_md_help_argument, translate_opts
        from onmt.translate.translator import build_translator
        import argparse
        parser = argparse.ArgumentParser(description='transformer-parser-base')
        add_md_help_argument(parser)
        translate_opts(parser, weights)
        opt = parser.parse_args(['-batch_size', '1'])
        translator = build_translator(opt, report_score=True, untrained=untrained)

        self._model_container = self.TransformerContainer(translator, opt)
        self.vocab_index = {word: index for index, word in
                            enumerate(self._model_container.translator.fields["src"].vocab.freqs)}
        index_vocab = {index: word for word, index in self.vocab_index.items()}
        self._model_container.index_vocab = index_vocab
        super(Transformer, self).__init__(
            identifier=self.identifier + ('-untrained' if untrained else ''),
            model=self._model_container, reset=lambda: None)  # transformer is feed-forward

    def __call__(self, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(*args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_last, **kwargs)
        elif self.mode == TaskModel.Modes.tokens_to_features:
            encodings = self._model_container(*args, **kwargs)
            # the onmt implementation concats things together, undo this
            return encodings[0].reshape(-1, self.features_size)

    class TransformerContainer:
        def __init__(self, translator, opt):
            self.translator = translator
            self.opt = opt
            self.index_vocab = None

        def __getattr__(self, name):
            return getattr(self.translator.model, name)

        def __call__(self, sentences):
            with tempfile.NamedTemporaryFile(mode='w+') as file:
                # separating sentences with newline, combined with a batch size of 1
                # will lead to one set of activations per sentence (albeit multiple words).
                if isinstance(sentences, np.ndarray) and not isinstance(sentences[0], str):
                    sentences = [" ".join([self.index_vocab[index] for index in sentences])]
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
    default_layers = [f'encoder.transformer.{i}.{layer}'
                      for i in range(6) for layer in ['feed_forward.layer_norm', 'feed_forward.dropout_2']]

    def tokenize(self, text, vocab_size=None):
        assert not vocab_size or vocab_size == self.vocab_size
        words = text.split()
        tokens = [self.vocab_index[word] for word in tqdm(words, desc='tokenize') if word in self.vocab_index]
        return np.array(tokens)

    @property
    def features_size(self):
        return 512  # encoding output of onmt transformer

    @property
    def vocab_size(self):
        return len(self._model_container.translator.fields["src"].vocab)


class _PytorchTransformerWrapper(BrainModel, TaskModel):
    def __init__(self, identifier, tokenizer, model, layers, sentence_average, tokenizer_special_tokens=()):
        super(_PytorchTransformerWrapper, self).__init__()
        self._logger = logging.getLogger(fullname(self))
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
        elif self.mode == TaskModel.Modes.tokens_to_features:
            return self._tokens_to_features(*args, **kwargs)
        elif self.mode == TaskModel.Modes.sentence_features:
            return self._sentence_features(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def _tokens_to_features(self, token_ids):
        import torch
        max_num_words = 512 - 2  # -2 for [cls], [sep]
        if os.getenv('ALLATONCE', '0') == '1':
            token_tensor = torch.tensor([token_ids])
            token_tensor = token_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
            features = self._model(token_tensor)[0][0]
            features = PytorchWrapper._tensor_to_numpy(features)
            return features
        features = []
        for token_index in range(len(token_ids)):
            context_start = max(0, token_index - max_num_words + 1)
            context_ids = token_ids[context_start:token_index + 1]
            tokens_tensor = torch.tensor([context_ids])
            tokens_tensor = tokens_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
            context_features = self._model(tokens_tensor)[0]
            features.append(PytorchWrapper._tensor_to_numpy(context_features[:, -1, :]))
        return np.concatenate(features)

    def _sentence_features(self, batch):
        import torch
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        if not any(self.identifier.startswith(prefix) for prefix in ['distilbert', 't5']):
            inputs["token_type_ids"] = (
                batch[2] if not any(self.identifier.startswith(prefix) for prefix in ["bert", "xlnet", "albert"])
                else None)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if self.identifier.startswith('t5'):
            # we already have a T5Wrapper in place that will convert input_ids to {encoder,decoder}_input_ids
            inputs['encoder_attention_mask'] = inputs['attention_mask']
            del inputs['attention_mask']
        with torch.no_grad():
            features_outputs = self._model(**inputs)
        # https://github.com/huggingface/transformers/blob/520e7f211926e07b2059bc8e21b668db4372e4db/src/transformers/modeling_bert.py#L811-L812
        sequence_output = features_outputs[0]
        if any(self.identifier.startswith(first_token_model) for first_token_model in
               ['bert', 'roberta', 'xlm', 'albert', 'distilbert', 'distilroberta']):
            # https://github.com/huggingface/transformers/blob/520e7f211926e07b2059bc8e21b668db4372e4db/src/transformers/modeling_bert.py#L454
            return sequence_output[:, 0]  # sentence features from first token (usually CLS)
        elif any(self.identifier.startswith(last_token_model) for last_token_model in
                 ['distilgpt2', 'openaigpt', 'gpt', 'xlnet', 'ctrl']):
            # use the last "real" token, ignoring padding by checking attention
            last_attended_token = []
            for batch in range(sequence_output.shape[0]):  # padding can be different per batch element
                attention = inputs['attention_mask'][batch]
                last_attention = torch.where(attention)[0].max()  # last index where attention is non-zero
                batch_token = sequence_output[batch, last_attention, :]
                last_attended_token.append(batch_token)
            last_attended_token = torch.stack(last_attended_token)
            return last_attended_token
        else:
            raise NotImplementedError(f"undefined if {self.identifier} should use "
                                      "first or last token for sentence features")

    @property
    def identifier(self):
        return self._extractor.identifier

    def tokenize(self, text, vocab_size=None):
        tokenized_text = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
        tokenized_text = np.array(tokenized_text)  # ~10 sec with numpy, ~40 hours without
        if (bool(vocab_size)) and vocab_size < self.vocab_size:  # smaller vocab size requested, drop tokens
            self._logger.debug(f"Shortening {self.vocab_size} to {vocab_size} (max in tokens: {max(tokenized_text)})")
            tokenized_text = np.array([token for token in tokenized_text if token < vocab_size])
        return tokenized_text

    def tokens_to_inputs(self, tokens):
        return np.array(self._tokenizer.build_inputs_with_special_tokens(tokens.tolist()))

    def glue_dataset(self, task, examples, label_list, output_mode, max_seq_length):
        import torch
        from torch.utils.data import TensorDataset
        from transformers import glue_convert_examples_to_features as convert_examples_to_features
        if task in ["mnli", "mnli-mm"] and \
                any(self.identifier.startswith(swap_model) for swap_model in ["roberta", "xlm-roberta"]):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        features = convert_examples_to_features(
            examples,
            self._tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(self.identifier.startswith("xlnet")),  # pad on the left for xlnet
            pad_token=self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token
                                                             if self._tokenizer.pad_token else ''])[0],
            pad_token_segment_id=4 if self.identifier.startswith("xlnet") else 0,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    @property
    def features_size(self):
        return self._model.config.hidden_size

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    def get_embedding_weights(self):
        modules = list(self._model.modules())
        while len(modules) > 1:  # 0th module is self
            modules = list(modules[1].modules())
        embedding_layer = modules[0]
        return embedding_layer.weight

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
                tokens = [
                    # tokens are sometimes padded by prefixes, clear those here
                    word.lstrip('##').lstrip('▁').rstrip('@@')
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


class KeyedVectorModel(BrainModel, TaskModel):
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
        self._vocab = self._model.vocab
        self._index2word_set = set(self._model.index2word)
        if random_embeddings:
            self._logger.debug(f"Replacing embeddings with random N(0, {random_std})")
            random_embedding = RandomState(0).randn(len(self._index2word_set), len(self._model['the'])) * random_std
            self._model = {word: random_embedding[i] for i, word in enumerate(sorted(self._index2word_set))}
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self._get_activations,
                                                     reset=lambda: None)
        self._extractor.insert_attrs(self)

    def __call__(self, stimuli, *args, average_sentence=True, **kwargs):
        if self.mode == BrainModel.Modes.recording:
            return _call_conditional_average(stimuli, *args, extractor=self._extractor,
                                             average_sentence=average_sentence, sentence_averaging=word_mean, **kwargs)
        elif self.mode == TaskModel.Modes.tokens_to_features:
            stimuli = " ".join(self._model.index2word[index] for index in stimuli)
            return self._encode_sentence(stimuli, *args, **kwargs)

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
        tokens = [self._vocab[word].index for word in text.split() if word in self._vocab
                  and self._vocab[word].index < vocab_size]  # only top-k vocab words
        return np.array(tokens)

    def glue_dataset(self, task, examples, label_list, output_mode, max_seq_length):
        import torch
        from torch.utils.data import TensorDataset
        tokens = [np.concatenate((self.tokenize(example.text_a),) +
                                 ((self.tokenize(example.text_b),) if example.text_b is not None else ()))
                  for example in examples]
        label_map = {label: i for i, label in enumerate(label_list)}
        labels = [label_map[label] for label in label_list]

        # Convert to Tensors and build dataset   - 3688 * 128
        # TODO: pad -- but how?
        token_tensors = torch.tensor(tokens, dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor(labels, dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(token_tensors, all_labels)
        return dataset

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def features_size(self):
        return 300


class Word2Vec(KeyedVectorModel):
    """
    Mikolov et al., 2013
    https://arxiv.org/pdf/1310.4546.pdf
    """

    identifier = 'word2vec'

    def __init__(self, weights_file='GoogleNews-vectors-negative300.bin', random_embeddings=False, **kwargs):
        weights_file = os.path.join(_ressources_dir, 'word2vec', weights_file)
        super(Word2Vec, self).__init__(
            identifier=self.identifier + ('-untrained' if random_embeddings else ''),
            weights_file=weights_file, binary=True,
            # standard embedding std
            # https://github.com/pytorch/pytorch/blob/ecbf6f99e6a4e373105133b31534c9fb50f2acca/torch/nn/modules/sparse.py#L106
            random_std=1, random_embeddings=random_embeddings, **kwargs)


class Glove(KeyedVectorModel):
    """
    Pennington et al., 2014
    http://www.aclweb.org/anthology/D14-1162
    """

    identifier = 'glove'

    def __init__(self, weights='glove.840B.300d.txt', random_embeddings=False, **kwargs):
        from gensim.scripts.glove2word2vec import glove2word2vec
        weights_file = os.path.join(_ressources_dir, 'glove', weights)
        word2vec_weightsfile = weights_file + '.word2vec'
        if not os.path.isfile(word2vec_weightsfile):
            glove2word2vec(weights_file, word2vec_weightsfile)
        super(Glove, self).__init__(
            identifier=self.identifier + ('-untrained' if random_embeddings else ''), weights_file=word2vec_weightsfile,
            # std from https://gist.github.com/MatthieuBizien/de26a7a2663f00ca16d8d2558815e9a6#file-fast_glove-py-L16
            random_std=.01, random_embeddings=random_embeddings, **kwargs)


class RecursiveNeuralTensorNetwork(BrainModel, TaskModel):
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
    WordPosition.identifier: LazyLoad(WordPosition),
    RandomEmbedding.identifier: LazyLoad(RandomEmbedding),
    SkipThoughts.identifier: LazyLoad(SkipThoughts),
    SkipThoughts.identifier + '-untrained': LazyLoad(lambda: SkipThoughts(load_weights=False)),
    LM1B.identifier: LazyLoad(LM1B),
    LM1B.identifier + '-untrained': LazyLoad(lambda: LM1B(reset_weights=True)),
    Word2Vec.identifier: LazyLoad(Word2Vec),
    Word2Vec.identifier + '-untrained': LazyLoad(lambda: Word2Vec(random_embeddings=True)),
    Glove.identifier: LazyLoad(Glove),
    Glove.identifier + '-untrained': LazyLoad(lambda: Glove(random_embeddings=True)),
    Transformer.identifier: LazyLoad(Transformer),
    Transformer.identifier + '-untrained': LazyLoad(lambda: Transformer(untrained=True)),
    ETM.identifier: LazyLoad(ETM),
    ETM.identifier + '-untrained': LazyLoad(lambda: ETM(random_embeddings=True)),
}
model_layers = {
    SentenceLength.identifier: SentenceLength.default_layers,
    WordPosition.identifier: WordPosition.default_layers,
    RandomEmbedding.identifier: RandomEmbedding.default_layers,
    SkipThoughts.identifier: SkipThoughts.default_layers,
    LM1B.identifier: LM1B.default_layers,
    Word2Vec.identifier: Word2Vec.default_layers,
    Glove.identifier: Glove.default_layers,
    Transformer.identifier: Transformer.default_layers,
    ETM.identifier: ETM.default_layers,
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

    def __call__(self, input_ids, **kwargs):
        # the decoder_input_ids are not right, but we only retrieve encoder features anyway
        return self._model(encoder_input_ids=input_ids, decoder_input_ids=input_ids, **kwargs)

    def __getattr__(self, item):  # forward attribute retrieval
        if item in ['_model', 'to']:
            return super(_T5Wrapper, self).__getattr__(item)
        return getattr(self._model, item)

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self


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
                # load standard model constructor: this will only create modules and initialize them for training
                model = model_ctr(config=config)
                state_dict = model.state_dict()  # capture initial random weights and force load them later
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
