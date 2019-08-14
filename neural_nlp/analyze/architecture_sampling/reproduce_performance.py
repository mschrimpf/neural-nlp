import os
import sys

import fire
import logging
import torch
from pytest import approx

import architecture_sampling
from architecture_sampling import utils
from architecture_sampling.evaluate import onmt
from architecture_sampling.evaluate.onmt import nmt_criterion
from architecture_sampling.evaluate.onmt.evaluate import evaluate_data
from neural_nlp.analyze.architecture_sampling import load_model, retrieve_log_value

_logger = logging.getLogger(__name__)


def run(model_dir, data):
    _logger.debug(f"Loading model from {model_dir}")
    model, params = load_model(model_dir, return_params=True)
    data_path = os.path.join(os.path.dirname(architecture_sampling.__file__), 'evaluate', 'data', data)
    _logger.debug(f"Loading data from {data_path}")
    dataset = torch.load(data_path)
    valid_data = onmt.Dataset(dataset['valid']['src'],
                              dataset['valid']['tgt'], params['batch_size'],
                              cuda=utils.cuda_available,
                              volatile=True)
    criterion = nmt_criterion(dataset['dicts']['tgt'].size())
    _logger.debug("Evaluating model")
    loss, accuracy = evaluate_data(model, criterion, valid_data, max_generator_batches=100)
    reported_loss = retrieve_log_value(model_dir, value_key='loss')
    assert loss == approx(reported_loss, abs=0.01), f"reproduced {loss} != previous {reported_loss}"


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    fire.Fire()
