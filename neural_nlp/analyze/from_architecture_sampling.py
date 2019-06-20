import json
import logging
import warnings

import torch
from pathlib import Path
from tqdm import tqdm

from architecture_sampling.evaluate.onmt import build_model
from brainscore.utils import LazyLoad
from neural_nlp import NaturalisticStoriesBenchmark, get_activations
from neural_nlp.models.wrapper.pytorch import PytorchWrapper
from result_caching import store

_logger = logging.getLogger(__name__)


def main(data_dir):
    data_dir = Path(data_dir)
    for model_dir in tqdm(data_dir.iterdir(), desc='models'):
        fin_file = model_dir / 'FIN'
        if not fin_file.is_file():
            warnings.warn(f"{model_dir} does not contain FIN file")
            continue
        score_model(model_dir)


@store()
def score_model(model_dir):
    fin_file = model_dir / 'FIN'
    text = fin_file.read_text().split('\n')
    params = json.loads(text[0])
    # checkpoint
    checkpoint_files = list(model_dir.glob("checkpoint-epoch*.pt"))
    checkpoint_file = max(checkpoint_files)
    checkpoint = torch.load(str(checkpoint_file), map_location=lambda storage, location: storage)

    params['encoder_architecture'] = params['model']
    params['decoder_architecture'] = None
    params['rnn_size'] = params['word_vec_size'] = params['hidden_size']
    params['layers'] = params['nlayers']
    del params['model'], params['hidden_size'], params['nlayers']
    del params['seed'], params['log_interval'], params['save'], params['independent_run'], \
        params['special_marker'], params['model_dir'], params['timestamp'], params['hostname'], \
        params['cuda_available'], params['hash']
    # training params
    batch_size = params['batch_size_eval']
    del params['data'], params['lr'], params['clip'], params['epochs'], params['batch_size'], params['bptt'], \
        params['beta'], params['tied'], params['batch_size_train'], params['batch_size_eval']
    # not sure about the following
    del params['embedding_size']
    params['source_size'], params['target_size'] = checkpoint['dicts']['src'].size(), checkpoint['dicts']['tgt'].size()
    decoder, encoder, generator, model = build_model(**params)

    # load weights
    _logger.info(f'Loading model from checkpoint at {checkpoint_file}')
    model.load_state_dict(checkpoint['model'])
    generator.load_state_dict(checkpoint['generator'])
    model.generator = generator
    layers = [f'encoder.rnn.cells.{cell}' for cell in [0,1]]

    activations_model = PytorchWrapper(identifier=model_dir, model=model, reset=lambda: model.reset(batch_size))

    def candidate(stimuli):
        return activations_model(stimuli, layers=layers)

    _logger.info('Running benchmark')
    benchmark = LazyLoad(NaturalisticStoriesBenchmark)
    score = benchmark(candidate)
    return score


if __name__ == '__main__':
    main('/braintree/data2/active/users/msch/zoo.bck20190408-multi30k/')
