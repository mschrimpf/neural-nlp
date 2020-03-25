import logging
from tqdm import tqdm

from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models import get_activations, model_layers, model_pool, SubsamplingHook
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store

_logger = logging.getLogger(__name__)


@store(identifier_ignore=['layers', 'prerun', 'model_impl'])
def score(benchmark, model, layers=None, model_impl=None, subsample=None):
    model_impl = model_impl or model_pool[model]
    if subsample:
        SubsamplingHook.hook(model, subsample)
    layers = layers or model_layers[model]

    _logger.info('Loading benchmark')
    benchmark_impl = benchmark_pool[benchmark]

    _logger.info('Running')
    # shortcut for performance benchmarks
    if any(benchmark.startswith(performance_prefix) for performance_prefix in ['wikitext', 'glue']):
        return benchmark_impl(model_impl)

    # only last layer for behavioral benchmarks
    if benchmark.startswith('Futrell2018'):
        layers = layers[-1:]

    layer_scores = []
    for i, layer in enumerate(tqdm(layers, desc='layers')):
        candidate = FixedLayer(model_impl, layer, prerun=layers if i == 0 else None)  # prerun everything for 1st layer
        layer_score = benchmark_impl(candidate)
        layer_score = layer_score.expand_dims('layer')
        layer_score['layer'] = [layer]
        layer_scores.append(layer_score)
    layer_scores = Score.merge(*layer_scores)
    layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    layer_scores.attrs['model'] = model
    layer_scores.attrs['benchmark'] = benchmark
    return layer_scores


class FixedLayer:
    def __init__(self, model, layer, prerun=None):
        self._model = model
        self._layer = layer
        self._prerun = prerun

    def __call__(self, *args, **kwargs):
        if self._prerun:  # avoid wasting computation: prerun all the layers to have them stored
            self._model(*args, **kwargs, layers=self._prerun)
        return self._model(*args, **kwargs, layers=[self._layer])

    def __getattr__(self, item):
        return self._model.__getattr__(item)

    def __setattr__(self, item, value):
        if item in ['_model', '_layer', '_prerun']:
            return super(FixedLayer, self).__setattr__(item, value)
        return self._model.__setattr__(item, value)
