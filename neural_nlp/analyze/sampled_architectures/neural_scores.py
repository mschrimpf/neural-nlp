import logging
import os
import sys
import warnings
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import fire
import itertools
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm

import architecture_sampling
from architecture_sampling import utils
from architecture_sampling.evaluate import onmt
from neural_nlp.analyze.sampled_architectures import load_model
from neural_nlp.analyze.sampled_architectures import retrieve_log_value
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models.wrapper.pytorch import PytorchWrapper
from result_caching import store

_logger = logging.getLogger(__name__)


def score_all_models(zoo_dir, benchmark='Pereira2018-encoding-min', perplexity_threshold=20):
    zoo_dir = Path(zoo_dir)
    model_dirs = list(zoo_dir.iterdir())
    scores = {}
    for model_dir in tqdm(model_dirs, desc='models'):
        try:
            perplexity = retrieve_log_value(model_dir)
            if perplexity_threshold and (np.isnan(perplexity) or perplexity > perplexity_threshold):
                _logger.debug(f"Ignoring {model_dir} due to poor perplexity ({perplexity})")
                continue
            _logger.debug(f"Scoring {model_dir} with ppl {perplexity}")
            score = _score_model(model_dir, benchmark=benchmark)
            scores[model_dir] = score
        except FileNotFoundError as e:
            _logger.warning(f"Ignoring {model_dir} due to {e}")
    return scores


@store(identifier_ignore=['sentences', 'index_dict'])
def prepare_stimulus_set(identifier, sentences, index_dict):
    """
    Convert a StimulusSet into input that can be fed to architecture-sampling models.
    Follows the steps in `architecture_sampling/setup.sh:45-56`.
    """
    with NamedTemporaryFile(suffix=f'-{identifier}.en') as sentences_file, \
            NamedTemporaryFile(suffix=f'-{identifier}.tok') as tokenized_file, \
            NamedTemporaryFile(suffix=f'-{identifier}.low') as lowercase_file:
        # write out
        _logger.debug(f"Writing stimuli to {sentences_file.name}")
        sentences_file.writelines([(sentence + os.linesep).encode() for sentence in sentences])
        sentences_file.flush()

        # tokenize
        _logger.debug(f"Tokenizing to {tokenized_file.name}")
        architecture_sampling_root = Path(architecture_sampling.__file__).parent
        tokenizer_path = architecture_sampling_root / "evaluate" / "data" / "onmt" / "tokenizer.perl"
        assert tokenizer_path.is_file()
        return_code = subprocess.call(["perl", str(tokenizer_path), "-no-escape", "-l", "en", "-q"],
                                      # pipe directly following https://stackoverflow.com/a/24209198/2225200
                                      # otherwise this call will hang
                                      stdin=open(sentences_file.name),
                                      stdout=open(tokenized_file.name, 'w'))
        assert return_code == 0

        # lower-case
        _logger.debug(f"Lower-casing to {lowercase_file.name}")
        lowercaser_path = architecture_sampling_root / "evaluate" / "data" / "onmt" / "lowercase.perl"
        assert lowercaser_path.is_file()
        return_code = subprocess.call(["perl", str(lowercaser_path)],
                                      stdin=open(tokenized_file.name),
                                      stdout=open(lowercase_file.name, 'w'))
        assert return_code == 0

        # preprocess
        preprocessed_stimuli = []
        # from architecture_sampling/evaluate/data/onmt/preprocess.py:118
        for line in lowercase_file.readlines():
            line = line.decode()
            embedding = index_dict.convertToIdx(line.split(), onmt.Constants.UNK_WORD)
            preprocessed_stimuli.append(embedding)
        return preprocessed_stimuli

    #  preprocess
    # for l in en de; do for f in data/multi30k/*.$l; do perl data/onmt/tokenizer.perl -no-escape -l $l -q < $f > $f.tok; done; done
    # for f in data/multi30k/*.tok; do perl data/onmt/lowercase.perl < $f > $f.low; done
    # PYTHONPATH=. python architecture_sampling/evaluate/data/onmt/preprocess.py \
    #     -train_src architecture_sampling/evaluate/data/multi30k/train.en.tok.low \
    #     -train_tgt architecture_sampling/evaluate/data/multi30k/train.de.tok.low \
    #     -valid_src architecture_sampling/evaluate/data/multi30k/val.en.tok.low \
    #     -valid_tgt architecture_sampling/evaluate/data/multi30k/val.de.tok.low \
    #     -test_src architecture_sampling/evaluate/data/multi30k/test.en.tok.low \
    #     -test_tgt architecture_sampling/evaluate/data/multi30k/test.de.tok.low \
    #     -save_data architecture_sampling/evaluate/data/multi30k.tok.low -lower


def score_model(model_dir, benchmark='Pereira2018-encoding-min'):
    score = _score_model(model_dir=model_dir, benchmark=benchmark)
    print(score.sel(aggregation='center'))


@store()
def _score_model(model_dir, benchmark='Pereira2018-decoding'):
    activations_model = build_activations_model(model_dir)
    _logger.info('Running benchmark')
    benchmark = benchmark_pool[benchmark]()
    score = benchmark(activations_model)
    return score


def build_activations_model(model_dir, load_weights=True):
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model dir {model_dir} does not exist")
    model, checkpoint, params = load_model(model_dir, load_weights=load_weights,
                                           return_checkpoint=True, return_params=True)
    layers = [f'encoder.rnn.cells.{cell}' for cell in [0, 1]]
    batch_size = 1  # params['batch_size']
    activations_model = ArchitectureWrapper(identifier=model_dir, model=MultiSentenceWrapper(model), layers=layers,
                                            reset=lambda: model.reset(batch_size), batch_size=batch_size,
                                            index_dict=checkpoint['dicts']['src'])  # use src dict since it's in English
    return activations_model


class ArchitectureWrapper(PytorchWrapper):
    def __init__(self, *args, index_dict, batch_size, layers, **kwargs):
        super(ArchitectureWrapper, self).__init__(*args, **kwargs)
        self._index_dict = index_dict
        self._batch_size = batch_size
        self._layers = layers
        self._current_stimuli_identifier = None

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            numpy_output = self._tensor_to_numpy(output)
            numpy_output = numpy_output[:_input[0].data.shape[1]]  # remove padding
            target_dict[name].append(numpy_output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def _tensor_to_numpy(self, output_hidden):
        full_hidden, (hidden_state, hidden_memory) = output_hidden
        hidden_state, hidden_memory = hidden_state.cpu().data.numpy(), hidden_memory.cpu().data.numpy()
        activations = np.concatenate((hidden_state, hidden_memory), axis=1)  # hiddens are batch x features
        return activations

    def __call__(self, stimuli, *args, **kwargs):
        if hasattr(stimuli, "name"):
            self._current_stimuli_identifier = stimuli.name
        result = super(ArchitectureWrapper, self).__call__(stimuli, *args, layers=self._layers, **kwargs)
        self._current_stimuli_identifier = None
        return result

    def get_activations(self, sentences, *args, **kwargs):
        _logger.debug(f"{len(sentences)} sentences")
        preprocessed_sentences = prepare_stimulus_set(identifier=self._current_stimuli_identifier, sentences=sentences,
                                                      index_dict=self._index_dict)
        num_unks = sum([encoding == self._index_dict.lookup(onmt.Constants.UNK_WORD)
                        for encoding in itertools.chain.from_iterable(preprocessed_sentences)])
        if num_unks:
            _logger.warning(f"{num_unks} unknowns in vocabulary for {self._current_stimuli_identifier}")
        data_sentences = onmt.Dataset(srcData=preprocessed_sentences, tgtData=None,
                                      batchSize=self._batch_size, cuda=utils.cuda_available, volatile=True)
        # the sentence-length in the preprocessed data is usually longer than sentence.split() due to '.', ',' etc.
        activations = super(ArchitectureWrapper, self).get_activations(data_sentences, *args, **kwargs)
        activations = OrderedDict((layer, np.concatenate(layer_activations))
                                  for layer, layer_activations in activations.items())
        return activations


class MultiSentenceWrapper:
    def __init__(self, model, propagate_hiddens=False):
        """
        :param propagate_hiddens: do not propagate hiddens between stories (at least valid for Pereira data)
        """
        self._model = model
        self._propagate_hiddens = propagate_hiddens

    def __call__(self, data_sentences):
        self._model.eval()
        hiddens = None
        num_batches = len(data_sentences)
        _logger.debug(f"Running {num_batches} batches")
        for num_batch in tqdm(range(num_batches), desc='sentence batches'):
            batch = data_sentences[num_batch]
            batch = batch[:-1]  # ignore indices following architecture_sampling/evaluate/onmt/evaluate.py:63-65
            hiddens = self._model(batch, enc_hidden=hiddens if self._propagate_hiddens else None, encoder_only=True)
            # the hiddens are supposed to be all zero because they are zero-ed for the decoder input (context is kept)

    def __getattr__(self, name):
        if name in ['_model']:
            return super(MultiSentenceWrapper, self).__getattr__(name)
        return getattr(self._model, name)  # this will not affect __call__ which is the only method we're providing


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()

# missing dependencies when running this with architecture-sampling interpreter:
# boto3 xarray peewee sklearn fire nltk "nltk_contrib @ git+https://github.com/nltk/nltk_contrib.git"
