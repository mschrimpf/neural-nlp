import argparse
import os
import pickle
from glob import iglob

import numpy as np
from keras.preprocessing import image


def main():
    models = {'densenet': densenet}
    parser = argparse.ArgumentParser('model comparison')
    parser.add_argument('--model', type=str, choices=[models.keys()], default='densenet')
    parser.add_argument('--layer', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--images_directory', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'images', 'sorted', 'Chairs'))
    args = parser.parse_args()
    print("Running with args", args)
    model, preprocess_input = models[args.model](args.image_size)
    model = use_model_layer(model, args.layer)
    model.summary()
    Y = {}
    image_filepaths = iglob(os.path.join(args.images_directory, '**', '*.png'), recursive=True)
    for image_filepath in image_filepaths:
        x = load_image(image_filepath, args.image_size)
        x = preprocess_input(x)
        preds = model.predict(x)
        Y[os.path.relpath(image_filepath, args.images_directory)] = preds
    save(Y, os.path.join(args.images_directory, 'activations'))


if __name__ == '__main__':
    main()


def load_image(image_filepath, image_size):
    img = image.load_img(image_filepath, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def save(image_activations, target_filepath):
    with open(target_filepath + '.pkl', 'wb') as file:
        pickle.dump(image_activations, file)


def use_model_layer(model, layer_name):
    if layer_name is None:
        return model
    last_layer_index = [layer.name for layer in model.layers].index(layer_name)
    for _ in range(len(model.layers) - last_layer_index - 1):
        model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    return model


def densenet(image_size):
    from DenseNet import DenseNetImageNet121, preprocess_input
    model = DenseNetImageNet121(input_shape=(image_size, image_size, 3))
    return model, preprocess_input
