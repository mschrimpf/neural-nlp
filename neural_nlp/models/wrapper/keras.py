import logging
from collections import OrderedDict

import numpy as np
from PIL import Image
from keras import backend as K
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201, preprocess_input as preprocess_densenet
from keras.applications.imagenet_utils import preprocess_input as preprocess_generic
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from keras.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.preprocessing import image
from keras_squeezenet import SqueezeNet

from candidate_models.models.implementations import DeepModel

_logger = logging.getLogger(__name__)


class KerasModel(DeepModel):
    def __init__(self, model_name, weights, batch_size, image_size):
        super(KerasModel, self).__init__(batch_size=batch_size, image_size=image_size)
        constructor, preprocessing = model_constructors_preprocessing[model_name]
        self._model = constructor(input_shape=(image_size, image_size, 3), weights=weights)
        self._preprocess_input = preprocessing

    def _load_image(self, image_filepath):
        img = image.load_img(image_filepath)
        x = image.img_to_array(img)
        return x

    def _preprocess_images(self, images, image_size):
        images = [self._preprocess_image(image, image_size) for image in images]
        return np.array(images)

    def _preprocess_image(self, img, image_size):
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((image_size, image_size))
        img = image.img_to_array(img)
        img = self._preprocess_input(img)
        return img

    def _get_activations(self, images, layer_names):
        input_tensor = self._model.input
        layers = [layer for layer in self._model.layers if layer.name in layer_names]
        layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
        layer_out_tensors = [layer.output for layer in layers]
        functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluate all tensors at once
        layer_outputs = functor([images, 0.])  # 0 to signal testing phase
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output
                            in zip([layer.name for layer in layers], layer_outputs)])

    def __repr__(self):
        return repr(self._model)


model_constructors_preprocessing = {
    'xception': (Xception, preprocess_xception),
    'densenet121': (DenseNet121, preprocess_densenet),  # https://arxiv.org/pdf/1608.06993.pdf
    'densenet169': (DenseNet169, preprocess_densenet),
    'densenet201': (DenseNet201, preprocess_densenet),
    'squeezenet': (SqueezeNet, preprocess_generic),
    'vgg-16': (VGG16, preprocess_vgg16),
    'vgg-19': (VGG19, preprocess_vgg19),
}
