import argparse
import os
import pickle
from glob import iglob

import numpy as np


def load_image(image_filepath, image_size):
    from keras.preprocessing import image
    img = image.load_img(image_filepath, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def save(image_activations, target_filepath):
    print("Saving to %s" % target_filepath)
    with open(target_filepath + '.pkl', 'wb') as file:
        pickle.dump(image_activations, file)


def get_model_outputs(model, x, layer_names):
    from keras import backend as K
    inp = model.input  # input placeholder
    selected_layers = [layer for layer in model.layers if layer.name in layer_names]
    outputs = [layer.output for layer in selected_layers]
    functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
    layer_outs = functor([x, 0.])  # 1.: training, 0.: test
    return list(zip([layer.name for layer in selected_layers], layer_outs))


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
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--images_directory', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'images', 'sorted', 'Chairs'))
    args = parser.parse_args()
    print("Running with args", args)

    model, preprocess_input = models[args.model](args.image_size)
    model.summary()
    assert all([layer in [l.name for l in model.layers] for layer in args.layers])

    Y = {}
    image_filepaths = iglob(os.path.join(args.images_directory, '**', '*.png'), recursive=True)
    for image_filepath in image_filepaths:
        print(image_filepath)
        x = load_image(image_filepath, args.image_size)
        x = preprocess_input(x)
        outputs = get_model_outputs(model, x, args.layers)
        Y[os.path.relpath(image_filepath, args.images_directory)] = outputs
    save(Y, os.path.join(args.images_directory, '%s-activations' % args.model))


if __name__ == '__main__':
    main()
