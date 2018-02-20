import logging
from enum import Enum

import keras
import torch

logger = logging.getLogger(__name__)


class ModelType(Enum):
    KERAS = 1
    PYTORCH = 2


def get_model_type(model):
    if isinstance(model, keras.engine.topology.Container):
        return ModelType.KERAS
    elif isinstance(model, torch.nn.Module):
        return ModelType.PYTORCH
    else:
        raise ValueError("Unsupported model framework: %s" % str(model))


PYTORCH_SUBMODULE_SEPARATOR = '.'
