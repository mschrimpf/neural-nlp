import logging

from brainio_base.assemblies import DataAssembly, walk_coords
from brainscore.metrics.rdm import RDM, RDMSimilarity
from brainscore.metrics.transformations import CartesianProduct, CrossValidation

from neural_nlp import models
from neural_nlp.models import get_activations, model_layers
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store_xarray

_logger = logging.getLogger(__name__)


def run(model, layers=None):
    layers = layers or model_layers[model]
    return _run(model=model, layers=layers)


@store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
def _run(model, layers):
    _logger.info('Loading neural data')
    neural_data = load_voxels()
    # leave-out ELvis story
    neural_data = neural_data[{'stimulus': [story != 'Elvis' for story in neural_data['story'].values]}]
    neural_data.attrs['stimulus_set'] = neural_data.attrs['stimulus_set'][
        [row.story != 'Elvis' for row in neural_data.attrs['stimulus_set'].itertuples()]]

    _logger.info('Computing activations')
    model_activations = get_activations(model_identifier=model, layers=layers, stimuli=neural_data.attrs['stimulus_set'])
    # since we're presenting all stimuli, including the inter-recording ones, we need to sub-filter
    model_activations = model_activations[{'stimulus': [sentence in neural_data['stimulus_sentence'].values
                                                        for sentence in model_activations['stimulus_sentence'].values]}]
    assert set(model_activations['stimulus_id'].values) == set(neural_data['stimulus_id'].values)

    _logger.info('Running benchmark')
    benchmark = MultiRegionBenchmark(target_assembly=neural_data)
    cross_layer = CartesianProduct(dividers=['layer'])
    scores = cross_layer(model_activations, apply=benchmark)
    return scores


class MultiRegionBenchmark:
    def __init__(self, target_assembly):
        self._target_assembly = target_assembly
        self._metric = RDMSimilarityCrossValidated()
        self._cross_region = CartesianProduct(dividers=['region'])

    def __call__(self, source_assembly):
        score = self._cross_region(self._target_assembly,
                                   apply=lambda region_assembly: self._metric(source_assembly, region_assembly))
        return score


class RDMSimilarityCrossValidated:
    # adapted from
    # https://github.com/dicarlolab/brain-score/blob/3d59d7a841fca63a5d346e599143f547560b5082/brainscore/metrics/rdm.py#L8

    class LeaveOneOutWrapper:
        def __init__(self, metric):
            self._metric = metric

        def __call__(self, train_source, train_target, test_source, test_target):
            # compare assemblies for a single split. we ignore the 10% train ("leave-one-out") and only use test.
            score = self._metric(test_source, test_target)
            return DataAssembly(score)

    def __init__(self, stimulus_coord='stimulus_sentence'):
        self._rdm = RDM()
        self._similarity = RDMSimilarity(comparison_coord=stimulus_coord)
        self._cross_validation = CrossValidation(test_size=.9,  # leave 10% out
                                                 split_coord=stimulus_coord, stratification_coord=None)

    def __call__(self, model_activations, target_rdm):
        model_activations = align(model_activations, target_rdm, on='stimulus_sentence')
        model_rdm = self._rdm(model_activations)
        leave_one_out = self.LeaveOneOutWrapper(self._similarity)
        # multi-dimensional coords with repeated dimensions not yet supported in CrossValidation
        drop_coords = [coord for coord, dims, value in walk_coords(target_rdm) if dims == ('stimulus', 'stimulus')]
        target_rdm = target_rdm.drop(drop_coords)
        return self._cross_validation(model_rdm, target_rdm, apply=leave_one_out)


def align(source, target, on):
    source_values, target_values = source[on].values.tolist(), target[on].values
    indices = [source_values.index(value) for value in target_values]
    assert len(source[on].dims) == 1, "multi-dimensional coordinates not implemented"
    dim = source[on].dims[0]
    dim_indices = {_dim: slice(None) if _dim != dim else indices for _dim in source.dims}
    aligned = source.isel(**dim_indices)
    return aligned
