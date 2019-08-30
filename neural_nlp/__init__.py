import logging

from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.transformations import apply_aggregate
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models import get_activations, model_layers, model_pool, SubsamplingHook
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['layers', 'prerun', 'model_impl'])
def score(benchmark, model, layers=None, model_impl=None, subsample=None, bold_shift=4):
    model_impl = model_impl or model_pool[model]
    if subsample:
        SubsamplingHook.hook(model, subsample)
    layers = layers or model_layers[model]

    _logger.info('Loading benchmark')
    benchmark = benchmark_pool[benchmark](bold_shift=bold_shift)
    if hasattr(benchmark, 'ceiling'):  # not yet implemented for all
        print(benchmark.ceiling)

    _logger.info('Running')
    layer_scores = []
    for layer in tqdm(layers, desc='layers'):
        candidate = lambda stimuli: model_impl(layers=[layer], stimuli=stimuli)
        layer_score = benchmark(candidate)
        layer_score = layer_score.expand_dims('layer')
        layer_score['layer'] = [layer]
        layer_scores.append(layer_score)
    layer_scores = Score.merge(*layer_scores)
    layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    score = apply_aggregate(lambda score: score.max('layer'), layer_scores)
    return score
