# from https://github.com/brain-score/model-tools

import logging

import functools
import numpy as np
from brainio_base.assemblies import NeuroidAssembly, walk_coords
from tqdm import tqdm

from brainscore.utils import fullname
from neural_nlp.stimuli import StimulusSet
from result_caching import store_xarray


class ActivationsExtractorHelper:
    def __init__(self, get_activations, reset, identifier=False):
        """
        :param identifier: an activations identifier for the stored results file. False to disable saving.
        :param reset: function to signal that a coherent stream of sentences is over.
        """
        self._logger = logging.getLogger(fullname(self))

        self.identifier = identifier
        self.get_activations = get_activations
        self.reset = reset
        self._stimulus_set_hooks = {}
        self._activations_hooks = {}

    def __call__(self, stimuli, layers, stimuli_identifier=None):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file. False to disable saving.
        """
        if isinstance(stimuli, StimulusSet):
            return self.from_stimulus_set(stimulus_set=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)
        else:
            return self.from_sentences(sentences=stimuli, layers=layers, stimuli_identifier=stimuli_identifier)

    def from_stimulus_set(self, stimulus_set, layers, stimuli_identifier=None):
        """
        :param stimuli_identifier: a stimuli identifier for the stored results file.
            False to disable saving. None to use `stimulus_set.name`
        """
        if stimuli_identifier is None:
            stimuli_identifier = stimulus_set.name
        for hook in self._stimulus_set_hooks.copy().values():  # copy to avoid stale handles
            stimulus_set = hook(stimulus_set)

        activations = self.from_sentences(sentences=stimulus_set['sentence'].values, layers=layers,
                                          stimuli_identifier=stimuli_identifier)
        activations = attach_stimulus_set_meta(activations, stimulus_set)
        return activations

    def from_sentences(self, sentences, layers, stimuli_identifier=None):
        assert layers is not None
        if self.identifier and stimuli_identifier:
            fnc = functools.partial(self._from_sentences_stored,
                                    identifier=self.identifier, stimuli_identifier=stimuli_identifier)
        else:
            self._logger.debug(f"self.identifier `{self.identifier}` or stimuli_identifier {stimuli_identifier} "
                               f"are not set, will not store")
            fnc = self._from_sentences
        return fnc(layers=layers, sentences=sentences)

    @store_xarray(identifier_ignore=['sentences', 'layers'], combine_fields={'layers': 'layer'})
    def _from_sentences_stored(self, identifier, layers, stimuli_identifier, sentences):
        return self._from_sentences(layers=layers, sentences=sentences)

    def _from_sentences(self, layers, sentences):
        self._logger.info('Running sentences')
        self.reset()
        layer_activations = self.get_activations(sentences, layers=layers)
        for hook in self._activations_hooks.copy().values():
            layer_activations = hook(layer_activations)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, sentences)

    def register_activations_hook(self, hook):
        r"""
        The hook will be called every time a batch of activations is retrieved.
        The hook should have the following signature::
            hook(batch_activations) -> batch_activations
        The hook should return new batch_activations which will be used in place of the previous ones.
        """

        handle = HookHandle(self._activations_hooks)
        self._activations_hooks[handle.id] = hook
        return handle

    def register_stimulus_set_hook(self, hook):
        r"""
        The hook will be called every time before a stimulus set is processed.
        The hook should have the following signature::
            hook(stimulus_set) -> stimulus_set
        The hook should return a new stimulus_set which will be used in place of the previous one.
        """

        handle = HookHandle(self._stimulus_set_hooks)
        self._stimulus_set_hooks[handle.id] = hook
        return handle

    def _package(self, layer_activations, sentences):
        shapes = [np.array(a).shape for a in layer_activations.values()]
        self._logger.debug('Activations shapes: {}'.format(shapes))
        self._logger.debug("Packaging individual layers")
        layer_assemblies = [self._package_layer(single_layer_activations, layer=layer, sentences=sentences) for
                            layer, single_layer_activations in tqdm(layer_activations.items(), desc='layer packaging')]
        # merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
        self._logger.debug("Merging layer assemblies")
        model_assembly = np.concatenate([a.values for a in layer_assemblies],
                                        axis=layer_assemblies[0].dims.index('neuroid'))
        nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
                             if set(dims) != {'neuroid'}}
        neuroid_coords = {coord: [dims, values] for coord, dims, values in walk_coords(layer_assemblies[0])
                          if set(dims) == {'neuroid'}}
        for layer_assembly in layer_assemblies[1:]:
            for coord in neuroid_coords:
                neuroid_coords[coord][1] = np.concatenate((neuroid_coords[coord][1], layer_assembly[coord].values))
            assert layer_assemblies[0].dims == layer_assembly.dims
            for dim in set(layer_assembly.dims) - {'neuroid'}:
                for coord, dims, values in walk_coords(layer_assembly[dim]):
                    assert (layer_assembly[coord].values == nonneuroid_coords[coord][1]).all()
        neuroid_coords = {coord: (dims_values[0], dims_values[1])  # re-package as tuple instead of list for xarray
                          for coord, dims_values in neuroid_coords.items()}
        model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
                                                   dims=layer_assemblies[0].dims)
        return model_assembly

    def _package_layer(self, layer_activations, layer, sentences):
        is_per_words = isinstance(layer_activations, list)
        if is_per_words:  # activations are retrieved per-word
            assert len(layer_activations) == 1 == layer_activations[0].shape[0] == len(sentences)
            activations = layer_activations[0][0]
            assert len(activations.shape) == 2
            words = sentences[0].split(' ')
            presentation_coords = {'stimulus_sentence': ('presentation', np.repeat(sentences, len(words))),
                                   'word': ('presentation', words)}
        else:  # activations are retrieved per-sentence
            assert layer_activations.shape[0] == len(sentences)
            activations = flatten(layer_activations)  # collapse for single neuroid dim
            presentation_coords = {'stimulus_sentence': ('presentation', sentences),
                                   'sentence_num': ('presentation', list(range(len(sentences))))}
        layer_assembly = NeuroidAssembly(
            activations,
            coords={**presentation_coords, **{
                'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
                'model': ('neuroid', [self.identifier] * activations.shape[1]),
                'layer': ('neuroid', [layer] * activations.shape[1]),
            }},
            dims=['presentation', 'neuroid']
        )
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
        layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
        return layer_assembly

    def insert_attrs(self, wrapper):
        wrapper.from_stimulus_set = self.from_stimulus_set
        wrapper.from_sentences = self.from_sentences
        wrapper.register_activations_hook = self.register_activations_hook
        wrapper.register_stimulus_set_hook = self.register_stimulus_set_hook


def attach_stimulus_set_meta(assembly, stimulus_set):
    sentences = stimulus_set['sentence'].values
    assert all(assembly['stimulus_sentence'].values == sentences)
    for column in stimulus_set.columns:
        if hasattr(assembly, column):
            continue
        assembly[column] = 'presentation', np.broadcast_to(stimulus_set[column].values, len(assembly['presentation']))
    return assembly


class HookHandle:
    next_id = 0

    def __init__(self, hook_dict):
        self.hook_dict = hook_dict
        self.id = HookHandle.next_id
        HookHandle.next_id += 1
        self._saved_hook = None

    def remove(self):
        hook = self.hook_dict[self.id]
        del self.hook_dict[self.id]
        return hook

    def disable(self):
        self._saved_hook = self.remove()

    def enable(self):
        self.hook_dict[self.id] = self._saved_hook
        self._saved_hook = None


def flatten(layer_output, return_index=False):
    flattened = layer_output.reshape(layer_output.shape[0], -1)
    if not return_index:
        return flattened

    def cartesian_product_broadcasted(*arrays):
        """
        http://stackoverflow.com/a/11146645/190597
        """
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        dtype = np.result_type(*arrays)
        rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
        out = np.empty(rows * cols, dtype=dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    index = cartesian_product_broadcasted(*[np.arange(s, dtype='int') for s in layer_output.shape[1:]])
    return flattened, index
