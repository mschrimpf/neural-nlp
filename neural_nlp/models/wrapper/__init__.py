import logging
from collections import OrderedDict

import numpy as np
from sklearn.decomposition import PCA

from brainscore.assemblies import NeuroidAssembly
from brainscore.utils import fullname


class Defaults(object):
    weights = 'imagenet'
    max_dimensionality = 1000


class DeepModel(object):
    """
    A model is defined by the model's name (defining the architecture) and its weights (defining the function).
    The model instantiation then relies on hyper-parameters `batch_size` and `max_dimensionality`
    where the specific choices should not change the results too much.
    Note that `max_dimensionality` might change the score of a value, but we are hoping
    that models with the same `max_dimensionality` at least remain comparable.
    """

    def __init__(self):
        # require arguments here to keep the signature of different implementations the same.
        # For instance, batch_size is not required for models other than TF but by requiring it here,
        # we keep the same method signature for the caller to simplify things.
        self._logger = logging.getLogger(fullname(self))

    def get_activations(self, sentences, layers,
                        max_dimensionality=Defaults.max_dimensionality):
        layer_activations = self._get_activations(sentences, layer_names=layers)
        self._pad_layers(layer_activations, num_components=max_dimensionality)
        self._logger.info('Reducing dimensionality')
        layer_activations = self._reduce_dimensionality(layer_activations, num_components=max_dimensionality)
        self._logger.info('Packaging into assembly')
        return self._package(layer_activations, sentences)

    def _get_activations(self, sentences, layer_names):
        raise NotImplementedError()

    def _reduce_dimensionality(self, layer_activations, num_components):
        def flatten(layer_output):
            return layer_output.reshape(layer_output.shape[0], -1)

        layer_activations = self._change_layer_activations(layer_activations, flatten)
        # return self._pca(layer_activations, num_components)
        return self._subsample(layer_activations, num_components)

    def _subsample(self, layer_activations, num_components):
        assert all(len(activations.shape) == 2 for activations in layer_activations.values()), \
            "all layer activations must be 2-dimensional (stimuli x flattened activations)"

        def subsample(activations):
            if activations.shape[1] <= num_components:
                return activations

            indices = np.random.randint(activations.shape[1], size=num_components)
            return activations[:, indices]

        return self._change_layer_activations(layer_activations, subsample)

    def _pca(self, layer_activations, num_components):
        assert all(len(activations.shape) == 2 for activations in layer_activations.values()), \
            "all layer activations must be 2-dimensional (stimuli x flattened activations)"

        if num_components is None:
            return layer_activations

        # compute components
        pca = self._change_layer_activations(layer_activations, lambda activations:
        PCA(n_components=num_components).fit(activations))

        # apply
        def reduce_dimensionality(layer_name, layer_activations):
            if layer_activations.shape[1] < num_components:
                self._logger.warning("layer {} activations are smaller than pca components: {}".format(
                    layer_name, layer_activations.shape))
                return layer_activations
            return pca[layer_name].transform(layer_activations)

        pca_activations = self._change_layer_activations(layer_activations, reduce_dimensionality, pass_name=True)
        return pca_activations

    def _package(self, layer_activations, sentences):
        activations = list(layer_activations.values())
        activations = np.array(activations)
        self._logger.debug('Activations shape: {}'.format(activations.shape))
        # layer x images x activations -> images x layer x activations
        activations = activations.transpose([1, 0, 2])
        assert activations.shape[0] == len(sentences)
        assert activations.shape[1] == len(layer_activations)
        layers = np.array(list(layer_activations.keys()))
        layers = np.repeat(layers[:, np.newaxis], repeats=activations.shape[-1], axis=1)
        activations = np.reshape(activations, [activations.shape[0], np.prod(activations.shape[1:])])
        layers = np.reshape(layers, [np.prod(activations.shape[1:])])
        model_assembly = NeuroidAssembly(
            activations,
            coords={'sentence': sentences,
                    'neuroid_id': ('neuroid', list(range(activations.shape[1]))),
                    'layer': ('neuroid', layers)},
            dims=['sentence', 'neuroid']
        )
        return model_assembly

    def _pad_layers(self, layer_activations, num_components):
        """
        make sure all layers are the minimum size
        """
        too_small_layers = [key for key, values in layer_activations.items()
                            if num_components is not None and values[0].size < num_components]
        for layer in too_small_layers:
            self._logger.warning("Padding layer {} with zeros since its activations are too small ({})".format(
                layer, layer_activations[layer].shape))
            layer_activations[layer] = [np.pad(a, (0, num_components - a.size), 'constant', constant_values=(0,))
                                        for a in layer_activations[layer]]

    def _change_layer_activations(self, layer_activations, change_function, pass_name=False):
        return OrderedDict((layer, change_function(values) if not pass_name else change_function(layer, values))
                           for layer, values in layer_activations.items())
