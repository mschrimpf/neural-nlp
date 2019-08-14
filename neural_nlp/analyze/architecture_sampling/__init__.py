import copy
import json
import logging
from pathlib import Path

import torch
from numpy.lib.arraysetops import _unpack_tuple

from architecture_sampling.evaluate.onmt import build_model

_logger = logging.getLogger(__name__)


def load_model(model_dir, return_checkpoint=False, return_params=False):
    model_dir = Path(model_dir)
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
    _logger.debug(f"Building model from params in {model_dir}")
    decoder, encoder, generator, model = _build_stored_model(checkpoint, params)
    # load weights
    _logger.info(f'Loading model from checkpoint at {checkpoint_file}')
    model.load_state_dict(checkpoint['model'])
    generator.load_state_dict(checkpoint['generator'])
    model.generator = generator
    output = (model,)
    if return_checkpoint:
        output += (checkpoint,)
    if return_params:
        output += (params,)
    return _unpack_tuple(output)


def _build_stored_model(checkpoint, params):
    params = copy.deepcopy(params)
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
    decoder, encoder, generator, model = build_model(**params)
    return decoder, encoder, generator, model
