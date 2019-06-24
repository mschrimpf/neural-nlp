import logging

from tqdm import tqdm

from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import VoxelBenchmark, fROIBenchmark, RDMBenchmark
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store_xarray

_logger = logging.getLogger(__name__)

benchmarks = {
    'voxel': VoxelBenchmark,
    'fROI': fROIBenchmark,
    'rdm': RDMBenchmark,
}


@store_xarray(identifier_ignore=['layers', 'prerun'], combine_fields={'layers': 'layer'})
def _run(benchmark, model, layers, prerun=True):
    _logger.info('Running benchmark')
    benchmark = benchmarks[benchmark]()
    if hasattr(benchmark, 'ceiling'):  # not yet implemented for all
        print(benchmark.ceiling)

    if prerun:
        get_activations(model_identifier=model, layers=layers, stimuli=benchmark._target_assembly.stimulus_set)

    layer_scores = []
    for layer in tqdm(layers, desc='layers'):
        candidate = lambda stimuli: get_activations(model_identifier=model, layers=[layer], stimuli=stimuli)
        layer_score = benchmark(candidate)
        layer_score = layer_score.expand_dims('layer')
        layer_score['layer'] = [layer]
        layer_scores.append(layer_score)
    layer_scores = Score.merge(*layer_scores)
    layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    return layer_scores
