import logging
from result_caching import store_xarray

from brainscore.assemblies import DataAssembly
from brainscore.benchmarks import SplitBenchmark
from brainscore.metrics import NonparametricWrapper
from brainscore.metrics.ceiling import SplitNoCeiling
from brainscore.metrics.rdm import RDMMetric
from neural_nlp import models
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data import load_rdm_sentences as load_neural_rdms

_logger = logging.getLogger(__name__)


class SourceRDMSimilarity(RDMMetric):
    def __call__(self, source_assembly, target_rdm):
        source_rdm = self._rdm(source_assembly)
        result = self._similarity(source_rdm, target_rdm)
        return DataAssembly(result)


def run(model, stimulus_set, layers=None):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers, stimulus_set=stimulus_set)


@store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'}, sub_fields=True)
def _run(model, layers, stimulus_set):
    _logger.info('Computing activations')
    model_activations = get_activations(model_name=model, layers=layers, stimulus_set_name=stimulus_set)

    _logger.info('Loading neural data')
    story = stimulus_set.split('.')[-1]
    neural_data = load_neural_rdms(story=story)
    neural_data = neural_data.mean(dim='subject')
    metric = NonparametricWrapper(SourceRDMSimilarity())
    primary_dimension, primary_coord = 'stimulus', 'stimulus_sentence'
    benchmark = SplitBenchmark(target_assembly=neural_data, metric=metric,
                               ceiling=SplitNoCeiling(), target_splits=['region'],
                               target_splits_kwargs=dict(non_dividing_dims=[primary_dimension]))

    _logger.info('Computing score')
    scores = benchmark(model_activations, transformation_kwargs=dict(
        alignment_kwargs=dict(order_dimensions=[primary_dimension], alignment_dim=primary_coord),
        cartesian_product_kwargs=dict(non_dividing_dims=[primary_dimension, 'neuroid'],
                                      dividing_coord_names_source=['layer']),
        cross_validation_kwargs=dict(dim=primary_coord, stratification_coord=None)
    ))
    return scores
