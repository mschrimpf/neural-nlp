import json
import logging
import os
import sys
import warnings
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import numpy as np
import subprocess
import torch
from pathlib import Path
from tqdm import tqdm

import architecture_sampling
from architecture_sampling import utils
from architecture_sampling.evaluate import onmt
from architecture_sampling.evaluate.onmt import build_model
from brainscore.utils import LazyLoad
from neural_nlp import PereiraDecoding
from neural_nlp.models.wrapper.pytorch import PytorchWrapper
from result_caching import store

_logger = logging.getLogger(__name__)


def _model_perplexity(model_dir):
    log_file = model_dir / 'log'
    if not log_file.exists():
        raise FileNotFoundError(f"Log file {log_file} does not exist")
    logs = log_file.read_text().split('\n')
    logs = [json.loads(log) for log in logs if log]
    val_ppls = [log for log in logs if log['data'] == 'valid']
    if not val_ppls:
        raise ValueError(f"No validation perplexities found in {log_file}")
    val_ppl = val_ppls[-1]
    return val_ppl['ppl']


def main(data_dir):
    data_dir = Path(data_dir)
    model_dirs = list(data_dir.iterdir())
    for model_dir in tqdm(model_dirs, desc='models'):
        try:
            perplexity = _model_perplexity(model_dir)
            if perplexity > 20:
                warnings.warn(f"Ignoring {model_dir} due to poor perplexity ({perplexity})")
                continue
            score = score_model(model_dir)
        except (FileNotFoundError, ValueError) as e:
            warnings.warn(f"Ignoring {model_dir} due to {e}")


@store(identifier_ignore=['stimuli', 'index_dict'])
def prepare_stimulus_set(identifier, stimuli, index_dict):
    """
    Convert a StimulusSet into input that can be fed to architecture-sampling models.
    Follows the steps in `setup.sh:45-56`.
    """
    with NamedTemporaryFile(suffix=f'-{stimuli.name}.en') as sentences_file, \
            NamedTemporaryFile(suffix=f'-{stimuli.name}.tok') as tokenized_file, \
            NamedTemporaryFile(suffix=f'-{stimuli.name}.low') as lowercase_file:
        # write out
        _logger.debug(f"Writing stimuli to {sentences_file.name}")
        sentences_file.writelines([(sentence + os.linesep).encode() for sentence in stimuli['sentence'].values])

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
            embedding = index_dict.convertToIdx(line, onmt.Constants.UNK_WORD)
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


@store()
def score_model(model_dir):
    fin_file = model_dir / 'FIN'
    if not fin_file.is_file():
        raise FileNotFoundError(f"No FIN file found in {model_dir}")
    text = fin_file.read_text().split('\n')
    params = json.loads(text[0])
    # checkpoint
    checkpoint_files = list(model_dir.glob("checkpoint-epoch*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    checkpoint_file = max(checkpoint_files)
    checkpoint = torch.load(str(checkpoint_file), map_location=lambda storage, location: storage)

    if 'encoder_architecture' not in params:
        params['encoder_architecture'] = params['model']
        del params['model']
    params['decoder_architecture'] = None
    params['rnn_size'] = params['word_vec_size'] = params['hidden_size']
    params['layers'] = params['nlayers']
    del params['hidden_size'], params['nlayers']
    del params['seed'], params['log_interval'], params['save'], params['independent_run'], \
        params['special_marker'], params['model_dir'], params['timestamp'], params['hostname'], \
        params['cuda_available'], params['hash']
    # clear training params
    batch_size = params['batch_size_eval'] if 'batch_size_eval' in params else params['batch_size']
    for training_param in ['data', 'lr', 'lr_decay', 'lr_start_decay_at', 'lr_cutoff', 'clip', 'epochs', 'bptt',
                           'beta', 'tied', 'batch_size', 'batch_size_train', 'batch_size_eval']:
        if training_param in params:
            del params[training_param]
    # not sure about the following
    del params['embedding_size']
    params['source_size'], params['target_size'] = checkpoint['dicts']['src'].size(), checkpoint['dicts']['tgt'].size()

    _logger.debug(f"Building model from params in {model_dir}")
    decoder, encoder, generator, model = build_model(**params)

    # load weights
    _logger.info(f'Loading model from checkpoint at {checkpoint_file}')
    model.load_state_dict(checkpoint['model'])
    generator.load_state_dict(checkpoint['generator'])
    model.generator = generator
    layers = [f'encoder.rnn.cells.{cell}' for cell in [0, 1]]

    batch_size = 1
    activations_model = ArchitectureWrapper(identifier=model_dir, model=MultiSentenceWrapper(model), layers=layers,
                                            reset=lambda: model.reset(batch_size), batch_size=batch_size,
                                            index_dict=checkpoint['dicts']['src'])  # use src dict since it's in English

    _logger.info('Running benchmark')
    benchmark = LazyLoad(PereiraDecoding)
    score = benchmark(activations_model)
    return score


class ArchitectureWrapper(PytorchWrapper):
    def __init__(self, *args, index_dict, batch_size, layers, **kwargs):
        super(ArchitectureWrapper, self).__init__(*args, **kwargs)
        self._index_dict = index_dict
        self._batch_size = batch_size
        self._layers = layers

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            numpy_output = self._tensor_to_numpy(output)
            target_dict[name].append(numpy_output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def _tensor_to_numpy(self, output_hidden):
        full_hidden, (hidden_state, hidden_memory) = output_hidden
        hidden_state, hidden_memory = hidden_state.cpu().data.numpy(), hidden_memory.cpu().data.numpy()
        activations = np.concatenate((hidden_state, hidden_memory), axis=1)  # hiddens are batch x features
        return activations

    def __call__(self, stimuli, *args, **kwargs):
        data_stimuli = prepare_stimulus_set(identifier=stimuli.name, stimuli=stimuli, index_dict=self._index_dict)
        data_stimuli = onmt.Dataset(srcData=data_stimuli, tgtData=None,
                                    batchSize=self._batch_size, cuda=utils.cuda_available, volatile=True)
        # TODO: check nans in data_stimuli
        # 8 dat2a_stimuli from 627 sentences (8 * 80 batch_size)
        return super(ArchitectureWrapper, self).__call__(data_stimuli, *args, layers=self._layers,
                                                         stimuli_identifier=stimuli.name, **kwargs)

    def get_activations(self, *args, **kwargs):
        activations = super(ArchitectureWrapper, self).get_activations(*args, **kwargs)
        activations = OrderedDict((layer, np.concatenate(layer_activations))
                                  for layer, layer_activations in activations.items())
        return activations


class MultiSentenceWrapper:
    def __init__(self, model, propagate_hiddens=False):
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

    def __getattr__(self, name):
        if name in ['_model']:
            return super(MultiSentenceWrapper, self).__getattr__(name)
        return getattr(self._model, name)  # this will not affect __call__ which is the only method we're providing


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main('/braintree/data2/active/users/msch/zoo.wmt17/')

# missing dependencies when running this with architecture-sampling interpreter:
# boto3 xarray peewee sklearn fire nltk "nltk_contrib @ git+https://github.com/nltk/nltk_contrib.git"
