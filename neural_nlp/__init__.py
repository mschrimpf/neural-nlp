import logging

from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.transformations import apply_aggregate
from neural_nlp import models
from neural_nlp.benchmarks import VoxelBenchmark, fROIBenchmark, RDMBenchmark, \
    PereiraDecoding
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store

_logger = logging.getLogger(__name__)

benchmarks = {
    'voxel': VoxelBenchmark,
    'fROI': fROIBenchmark,
    'rdm': RDMBenchmark,
    'Pereira2018-decoding': PereiraDecoding,
}


@store(identifier_ignore=['layers', 'prerun'])
def _run(benchmark, model, layers, subsample=None, prerun=True, bold_shift=4):
    _logger.info('Running benchmark')
    benchmark = benchmarks[benchmark](bold_shift=bold_shift)
    if hasattr(benchmark, 'ceiling'):  # not yet implemented for all
        print(benchmark.ceiling)

    if prerun:
        stimulus_set = benchmark._target_assembly.stimulus_set
        get_activations(model_identifier=model, layers=layers, subsample=subsample,
                        stimuli=stimulus_set, stimuli_identifier=stimulus_set.name)

    layer_scores = []
    for layer in tqdm(layers, desc='layers'):
        candidate = lambda stimuli: get_activations(
            model_identifier=model, layers=[layer], subsample=subsample, stimuli=stimuli)
        layer_score = benchmark(candidate)
        layer_score = layer_score.expand_dims('layer')
        layer_score['layer'] = [layer]
        layer_scores.append(layer_score)
    layer_scores = Score.merge(*layer_scores)
    layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    score = apply_aggregate(lambda score: score.max('layer'), layer_scores)
    return score
