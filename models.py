import argparse
import functools
import logging
import os
import sys
from collections import OrderedDict
from enum import Enum
from glob import iglob

import keras
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.autograd import Variable

from utils import save

PYTORCH_SUBMODULE_SEPARATOR = '.'


def densenet(image_size):
    from DenseNet import DenseNetImageNet121, preprocess_input
    model = DenseNetImageNet121(input_shape=(image_size, image_size, 3))
    return model, preprocess_input


def squeezenet(image_size):
    from keras_squeezenet import SqueezeNet
    from keras.applications.imagenet_utils import preprocess_input
    model = SqueezeNet(weights='imagenet', input_shape=(image_size, image_size, 3))
    return model, preprocess_input


def vgg16(image_size):
    from keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(weights='imagenet', input_shape=(image_size, image_size, 3))
    return model, preprocess_input


def marrnet(image_size):
    from marrnet import MarrNet, preprocess
    model = MarrNet()
    if image_size != 256:
        logger.warning("Image size will be set to 256")
    return model, preprocess()


logger = logging.getLogger()


def main():
    models = {
        'vgg16': vgg16,
        'densenet': densenet,
        'squeezenet': squeezenet,
        'marrnet': marrnet
    }
    parser = argparse.ArgumentParser('model comparison')
    parser.add_argument('--model', type=str, choices=list(models.keys()), default='squeezenet')
    parser.add_argument('--layers', nargs='+', default=None)
    parser.add_argument('--pca', type=int, default=200,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--images_directory', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'images', 'sorted'))
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger.info("Running with args %s", vars(args))

    # model
    logger.debug('Creating model')
    model, preprocess_input = models[args.model](args.image_size)
    print_verify_model(model, args.layers)
    model_type = get_model_type(model)

    # input
    logger.debug('Loading input images')
    image_filepaths, images = prepare_images(args.images_directory, args.image_size, preprocess_input, model_type)

    # output
    logger.debug('Computing activations')
    layer_outputs = get_model_outputs(model, images, args.layers, pca_components=args.pca)
    Y = {}
    for i, image_filepath in enumerate(image_filepaths):
        image_relpath = os.path.relpath(image_filepath, args.images_directory)
        Y[image_relpath] = {layer_name: layer_outputs[i] for layer_name, layer_outputs in layer_outputs.items()}
    save({'activations': Y, 'args': args}, os.path.join(args.images_directory, '%s-activations' % args.model))


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


def print_verify_model(model, layer_names):
    model_type = get_model_type(model)
    _print_verify_model = {ModelType.KERAS: print_verify_model_keras,
                           ModelType.PYTORCH: print_verify_model_pytorch}[model_type]
    _print_verify_model(model, layer_names)


def print_verify_model_pytorch(model, layer_names):
    logger.info(str(model))

    def collect_pytorch_layer_names(module, parent_module_parts):
        result = []
        for submodule_name, submodule in module._modules.items():
            if not isinstance(submodule, torch.nn.Sequential):
                result.append(PYTORCH_SUBMODULE_SEPARATOR.join(parent_module_parts + [submodule_name]))
            else:
                result += collect_pytorch_layer_names(submodule, parent_module_parts + [submodule_name])
        return result

    nonexisting_layers = set(layer_names) - set(collect_pytorch_layer_names(model, []))
    assert len(nonexisting_layers) == 0, "Layers not found in PyTorch model: %s" % str(nonexisting_layers)


def print_verify_model_keras(model, layer_names):
    model.summary(print_fn=logger.info)
    nonexisting_layers = set(layer_names) - set([layer.name for layer in model.layers])
    assert len(nonexisting_layers) == 0, "Layers not found in keras model: %s" % str(nonexisting_layers)


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


def get_model_outputs(model, x, layer_names, pca_components=None):
    logger.info('Computing layer outputs')
    model_type = get_model_type(model)
    compute_layer_outputs = {ModelType.KERAS: compute_layer_outputs_keras,
                             ModelType.PYTORCH: compute_layer_outputs_pytorch}[model_type]
    return compute_layer_outputs(layer_names, model, x,
                                 functools.partial(arrange_layer_output, pca_components=pca_components))


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
        layer.register_forward_hook(lambda _layer, _input, output: store_layer_output(layer_name, output))
    model(x)
    return layer_results


def arrange_layer_output(layer_output, pca_components):
    if pca_components is not None and np.prod(layer_output.shape[1:]) > pca_components:
        layer_output = layer_output.reshape(layer_output.shape[0], -1)
        layer_output = PCA(n_components=pca_components).fit_transform(layer_output)
    return layer_output


if __name__ == '__main__':
    main()
