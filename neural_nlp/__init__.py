import logging

from brainscore.assemblies import DataAssembly
from brainscore.metrics.rdm import RDM, RDMSimilarity
from brainscore.metrics.transformations import CartesianProduct
from result_caching import store_xarray

from neural_nlp import models
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data import load_rdm_sentences as load_neural_rdms

_logger = logging.getLogger(__name__)


class MultiRegionBenchmark:
    def __init__(self, target_assembly):
        self._target_assembly = target_assembly
        self._metric = HalfRDMSimilarity()
        self._cross_region = CartesianProduct(dividers=['region'])

    def __call__(self, source_assembly):
        score = self._cross_region(self._target_assembly,
                                   apply=lambda region_assembly: self._metric(source_assembly, region_assembly))
        return score


class HalfRDMSimilarity:
    def __init__(self, stimulus_coord='stimulus_sentence'):
        self._rdm = RDM()
        self._similarity = RDMSimilarity(comparison_coord=stimulus_coord)

    def __call__(self, model_activations, target_rdm):
        model_activations = align(model_activations, target_rdm, on='stimulus_sentence')
        model_rdm = self._rdm(model_activations)
        similarity = self._similarity(model_rdm, target_rdm)
        return DataAssembly(similarity)


def run(model, stimulus_set, layers=None):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers, stimulus_set=stimulus_set)


@store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
def _run(model, layers, stimulus_set):
    _logger.info('Computing activations')
    model_activations = get_activations(model_name=model, layers=layers, stimulus_set_name=stimulus_set)

    _logger.info('Loading neural data')
    story = stimulus_set.split('.')[-1]
    neural_data = load_neural_rdms(story=story)
    neural_data = neural_data.mean(dim='subject')

    _logger.info('Running benchmark')
    benchmark = MultiRegionBenchmark(target_assembly=neural_data)
    cross_layer = CartesianProduct(dividers=['layer'])
    scores = cross_layer(model_activations, apply=benchmark)
    return scores


def align(source, target, on):
    source_values, target_values = source[on].values.tolist(), target[on].values
    indices = [source_values.index(value) for value in target_values]
    assert len(source[on].dims) == 1, "multi-dimensional coordinates not implemented"
    dim = source[on].dims[0]
    dim_indices = {_dim: slice(None) if _dim != dim else indices for _dim in source.dims}
    aligned = source.isel(**dim_indices)
    return aligned
