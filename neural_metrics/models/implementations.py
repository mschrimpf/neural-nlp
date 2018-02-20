import functools
import logging
import os
from glob import iglob

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from neural_metrics.models.type import ModelType

_logger = logging.getLogger(__name__)


def densenet(image_size, weights='imagenet'):
    from DenseNet import DenseNetImageNet121, preprocess_input
    model = DenseNetImageNet121(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def squeezenet(image_size, weights='imagenet'):
    from keras_squeezenet import SqueezeNet
    from keras.applications.imagenet_utils import preprocess_input
    model = SqueezeNet(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def vgg16(image_size, weights='imagenet'):
    from keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


def mobilenet(image_size, weights='imagenet'):
    from keras.applications.mobilenet import MobileNet, preprocess_input
    model = MobileNet(input_shape=(image_size, image_size, 3), weights=weights)
    return model, preprocess_input


model_mappings = {
    'vgg16': vgg16,
    'densenet': densenet,
    'squeezenet': squeezenet,
    'mobilenet': mobilenet
}


def prepare_images(images_directory, image_size, preprocess_input, model_type):
    image_filepaths = iglob(os.path.join(images_directory, '**', '*.png'), recursive=True)
    image_filepaths = list(image_filepaths)
    load_image = {ModelType.KERAS: functools.partial(load_image_keras, image_size=image_size),
                  ModelType.PYTORCH: Image.open}[model_type]
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = [preprocess_input(image) for image in images]
    concat = {ModelType.KERAS: np.concatenate,
              ModelType.PYTORCH: lambda _images: Variable(torch.cat(_images))}[model_type]
    images = concat(images)
    return image_filepaths, images


def load_image_keras(image_filepath, image_size):
    from keras.preprocessing import image
    img = image.load_img(image_filepath, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x
