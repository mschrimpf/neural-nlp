import logging
import os

import caching
from caching import store_xarray

from brainscore.assemblies import DataAssembly
from brainscore.benchmarks import SplitBenchmark
from brainscore.metrics import NonparametricWrapper, Metric
from brainscore.metrics.ceiling import SplitNoCeiling
from brainscore.metrics.rdm import RDMSimilarity, RDM
from neural_nlp import models
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data import load_rdm_sentences as load_neural_rdms

caching.store.configure_storagedir(os.path.join(os.path.dirname(__file__), '..', 'output'))

_logger = logging.getLogger(__name__)


class SimilarityWrapper(Metric):
    def __init__(self, metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric = metric

    def __call__(self, source_assembly, target_assembly):
        result = self._metric(source_assembly, target_assembly)
        return DataAssembly(result)


def run(model, stimulus_set, layers=None):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers, stimulus_set=stimulus_set)


@store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
def _run(model, layers, stimulus_set):
    _logger.info('Computing activations')
    model_activations = get_activations(model_name=model, layers=layers, stimulus_set_name=stimulus_set)
    model_activations = RDM()(model_activations)

    _logger.info('Loading neural data')
    story = stimulus_set.split('.')[-1]
    neural_data = load_neural_rdms(story=story)
    neural_data = neural_data.mean(dim='subject')
    similarity = NonparametricWrapper(SimilarityWrapper(RDMSimilarity()))
    primary_dimensions = ('stimulus',)
    benchmark = SplitBenchmark(target_assembly=neural_data, metric=similarity,
                               ceiling=SplitNoCeiling(), target_splits=['region'],
                               target_splits_kwargs=dict(non_dividing_dims=primary_dimensions))

    _logger.info('Computing score')
    scores = benchmark(model_activations, transformation_kwargs=dict(
        alignment_kwargs=dict(order_dimensions=primary_dimensions, alignment_dim='stimulus_sentence'),
        cartesian_product_kwargs=dict(non_dividing_dims=primary_dimensions),
        cross_validation_kwargs=dict(dim='stimulus_sentence', stratification_coord=None)
    ))
    return scores
