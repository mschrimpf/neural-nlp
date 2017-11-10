import argparse
import os
from collections import OrderedDict
from glob import iglob

import logging
import numpy as np
import sys
from sklearn.decomposition import PCA

import keras
import torch
from utils import save


def load_image(image_filepath, image_size):
    from keras.preprocessing import image
    img = image.load_img(image_filepath, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def compute_layer_outputs_keras(layer_names, model, x):
    from keras import backend as K
    input_tensor = model.input
    layers = [layer for layer in model.layers if layer.name in layer_names]
    layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
    layer_out_tensors = [layer.output for layer in layers]
    functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluation all tensors
    layer_outputs = functor([x, 0.])  # 1.: training, 0.: test
    return OrderedDict([(layer_name, layer_output) for layer_name, layer_output
                        in zip([layer.name for layer in layers], layer_outputs)])


def compute_layer_outputs_pytorch(layer_names, model, x):
    layer_results = OrderedDict()

    def store_layer_output(layer_name, output):
        layer_results[layer_name] = output.data.numpy()

    for layer_name in layer_names:
        layer = model.get(layer_name)
        layer.register_forward_hook(lambda _layer, _input, output: store_layer_output(layer_name, output))
    model(x)
    return layer_results


def arrange_layer_outputs(layer_results, pca_components):
    for layer_name, layer_output in layer_results.items():
        if pca_components is not None and np.prod(layer_output.shape[1:]) > pca_components:
            layer_output = layer_output.reshape(layer_output.shape[0], -1)
            layer_output = PCA(n_components=pca_components).fit_transform(layer_output)
            layer_results[layer_name] = layer_output
    return layer_results


def get_model_outputs(model, x, layer_names, pca_components=None):
    logger.info('Computing model\'s layer outputs')
    if isinstance(model, keras.engine.topology.Container):
        layer_results = compute_layer_outputs_keras(layer_names, model, x)
    elif isinstance(model, torch.nn.Module):
        layer_results = compute_layer_outputs_pytorch(layer_names, model, x)
    else:
        raise ValueError("Unsupported model framework: %s" % str(model))
    return arrange_layer_outputs(layer_results, pca_components)


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


def main():
    models = {
        'vgg16': vgg16,
        'densenet': densenet,
        'squeezenet': squeezenet
    }
    parser = argparse.ArgumentParser('model comparison')
    parser.add_argument('--model', type=str, choices=list(models.keys()), default='squeezenet')
    parser.add_argument('--layers', nargs='+', default=None)
    parser.add_argument('--pca', type=int, default=200,
                        help='Number of components to reduce the flattened features to')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--images_directory', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'images', 'sorted', 'Chairs'))
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger.info("Running with args %s", vars(args))

    # model
    logger.debug('Creating model')
    model, preprocess_input = models[args.model](args.image_size)
    model.summary()
    assert all([layer in [l.name for l in model.layers] for layer in args.layers])

    # input
    logger.debug('Loading input images')
    image_filepaths = iglob(os.path.join(args.images_directory, '**', '*.png'), recursive=True)
    image_filepaths = list(image_filepaths)
    images = [load_image(image_filepath, args.image_size) for image_filepath in image_filepaths]
    images = [preprocess_input(image) for image in images]
    images = np.concatenate(images)

    # output
    logger.debug('Computing activations')
    layer_outputs = get_model_outputs(model, images, args.layers, pca_components=args.pca)
    Y = {}
    for i, image_filepath in enumerate(image_filepaths):
        image_relpath = os.path.relpath(image_filepath, args.images_directory)
        Y[image_relpath] = {layer_name: layer_outputs[i] for layer_name, layer_outputs in layer_outputs.items()}
    save({'activations': Y, 'args': args}, os.path.join(args.images_directory, '%s-activations' % args.model))


logger = logging.getLogger()

if __name__ == '__main__':
    main()
