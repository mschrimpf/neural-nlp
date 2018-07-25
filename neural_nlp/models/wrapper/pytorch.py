import logging
from collections import OrderedDict

import torch
from torch.autograd import Variable

from . import DeepModel

_logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.WARNING)

SUBMODULE_SEPARATOR = '.'


class PytorchModel(DeepModel):
    def __init__(self):
        super().__init__()
        self._model = self._load_model()
        if torch.cuda.is_available():
            self._model.cuda()

    def _load_model(self):
        raise NotImplementedError()

    def _get_activations(self, sentences, layer_names):
        layer_results = OrderedDict()

        def walk_pytorch_module(module, layer_name):
            for part in layer_name.split(SUBMODULE_SEPARATOR):
                module = module._modules.get(part)
            return module

        def store_layer_output(layer_name, output):
            layer_results[layer_name] = output.data.numpy()

        for layer_name in layer_names:
            layer = walk_pytorch_module(self._model, layer_name)
            layer.register_forward_hook(
                lambda _layer, _input, output, name=layer_name: store_layer_output(name, output))
        sentences = Variable(sentences)
        if torch.cuda.is_available():
            sentences.cuda()
        self._model(sentences)
        return layer_results

    def __repr__(self):
        return repr(self._model)

    def _collect_layers(self, module):
        layers = []
        for submodule_name, submodule in module._modules.items():
            if not submodule._modules:
                sublayers = [submodule_name]
            else:
                sublayers = self._collect_layers(submodule)
                sublayers = [submodule_name + SUBMODULE_SEPARATOR + layer for layer in sublayers]
            layers += sublayers
        return layers

    def available_layers(self):
        return self._collect_layers(self._model)
