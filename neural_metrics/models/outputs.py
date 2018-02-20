import copy
import functools
import logging
from collections import OrderedDict

import numpy as np
from sklearn.decomposition import PCA

from neural_metrics.models.type import get_model_type, ModelType, PYTORCH_SUBMODULE_SEPARATOR

_logger = logging.getLogger(__name__)


def get_model_outputs(model, x, layer_names, batch_size=None, pca_components=None):
    _logger.info('Computing layer outputs')
    model_type = get_model_type(model)
    compute_layer_outputs = {ModelType.KERAS: compute_layer_outputs_keras,
                             ModelType.PYTORCH: compute_layer_outputs_pytorch}[model_type]
    if batch_size is None or not (0 < batch_size < len(x)):
        _logger.debug("Computing all outputs at once")
        return compute_layer_outputs(layer_names, model, x,
                                     functools.partial(arrange_layer_output, pca_components=pca_components))

    outputs = None
    batch_start = 0
    while batch_start < len(x):
        batch_end = min(batch_start + batch_size, len(x))
        _logger.debug('Batch: %d->%d/%d', batch_start, batch_end, len(x))
        batch = x[batch_start:batch_end]
        batch_output = compute_layer_outputs(layer_names, model, batch)
        if outputs is None:
            outputs = copy.copy(batch_output)
        else:
            for layer_name, layer_output in batch_output.items():
                outputs[layer_name] = np.concatenate((outputs[layer_name], layer_output))
        batch_start = batch_end
    for layer_name, layer_output in outputs.items():
        _logger.debug('Arranging layer output %s (shape %s)', layer_name, str(layer_output.shape))
        outputs[layer_name] = arrange_layer_output(layer_output, pca_components=pca_components)
    return outputs


def compute_layer_outputs_keras(layer_names, model, x, arrange_output=lambda x: x):
    from keras import backend as K
    input_tensor = model.input
    layers = [layer for layer in model.layers if layer.name in layer_names]
    layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
    layer_out_tensors = [layer.output for layer in layers]
    functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluate all tensors at once
    layer_outputs = functor([x, 0.])  # 1.: training, 0.: test
    return OrderedDict([(layer_name, arrange_output(layer_output)) for layer_name, layer_output
                        in zip([layer.name for layer in layers], layer_outputs)])


def compute_layer_outputs_pytorch(layer_names, model, x, arrange_output=lambda x: x):
    layer_results = OrderedDict()

    def walk_pytorch_module(module, layer_name):
        for part in layer_name.split(PYTORCH_SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
        return module

    def store_layer_output(layer_name, output):
        layer_results[layer_name] = arrange_output(output.data.numpy())

    for layer_name in layer_names:
        layer = walk_pytorch_module(model, layer_name)
        layer.register_forward_hook(lambda _layer, _input, output, name=layer_name: store_layer_output(name, output))
    model(x)
    return layer_results


def arrange_layer_output(layer_output, pca_components):
    if pca_components is not None and 0 < pca_components < np.prod(layer_output.shape[1:]):
        assert layer_output.shape[0] >= pca_components, \
            "output has %d components but must have more than %d PCA components" % (
                layer_output.shape[0], pca_components)
        layer_output = layer_output.reshape(layer_output.shape[0], -1)
        layer_output = PCA(n_components=pca_components).fit_transform(layer_output)
    return layer_output
