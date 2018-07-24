import glob
import logging
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import tensorflow as tf
from nets import nets_factory
from preprocessing import inception_preprocessing, vgg_preprocessing

from candidate_models.models.implementations import DeepModel, Defaults

_logger = logging.getLogger(__name__)

slim_models = pd.read_csv(os.path.join(os.path.dirname(__file__), 'models.csv'))
slim_models = slim_models[slim_models['framework'] == 'slim']


class TensorflowSlimModel(DeepModel):
    def __init__(self, model_name, weights=Defaults.weights,
                 batch_size=Defaults.batch_size, image_size=Defaults.image_size):
        super().__init__(batch_size=batch_size, image_size=image_size)
        self._create(model_name, batch_size, image_size)
        self._sess = tf.Session()
        self._restore(model_name, weights)

    def _create(self, model_name, batch_size, image_size):
        _model_properties = self._get_model_properties(model_name)
        call = _model_properties['callable']
        arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0.)
        kwargs = {}
        if model_name.startswith('mobilenet_v2') or model_name.startswith('mobilenet_v1'):
            arg_scope = nets_factory.arg_scopes_map[call](weight_decay=0., is_training=False)
            kwargs = {'depth_multiplier': _model_properties['depth_multiplier']}
            call = 'mobilenet_v2' if model_name.startswith('mobilenet_v2') else 'mobilenet_v1'
        tf.reset_default_graph()
        model = nets_factory.networks_map[call]
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
        preprocess_image = vgg_preprocessing.preprocess_image if _model_properties['preprocessing'] == 'vgg' \
            else inception_preprocessing.preprocess_image
        self.inputs = tf.map_fn(lambda image: preprocess_image(tf.image.convert_image_dtype(image, dtype=tf.uint8),
                                                               image_size, image_size), self.inputs)
        with tf.contrib.slim.arg_scope(arg_scope):
            logits, self.endpoints = model(self.inputs,
                                           num_classes=1001 - int(_model_properties['labels_offset']),
                                           is_training=False,
                                           **kwargs)

    def _get_model_properties(self, model_name):
        _model_properties = slim_models[slim_models['model'] == model_name]
        _model_properties = {field: next(iter(_model_properties[field]))
                             for field in _model_properties.columns}
        return _model_properties

    def _get_model_path(self):
        search_paths = ['/braintree/data2/active/users/qbilius/models/slim',
                        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model-weights')]
        for search_path in search_paths:
            if os.path.isdir(search_path):
                self._logger.debug("Using model path '{}'".format(search_path))
                return search_path
        raise ValueError("No model path found in {}".format(search_paths))

    def _restore(self, model_name, weights):
        assert weights == 'imagenet'
        var_list = None
        if model_name.startswith('mobilenet'):
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy
            ema = tf.train.ExponentialMovingAverage(0.999)
            var_list = ema.variables_to_restore()
        restorer = tf.train.Saver(var_list)

        model_path = self._get_model_path()
        fnames = glob.glob(os.path.join(model_path, model_name, '*.ckpt*'))
        assert len(fnames) > 0
        restore_path = fnames[0].split('.ckpt')[0] + '.ckpt'
        restorer.restore(self._sess, restore_path)

    def _load_image(self, image_filepath):
        image = skimage.io.imread(image_filepath)
        return image

    def _preprocess_images(self, images, image_size):
        images = [self._preprocess_image(image, image_size) for image in images]
        return np.array(images)

    def _preprocess_image(self, image, image_size):
        image = skimage.transform.resize(image, (image_size, image_size))
        assert image.min() >= 0
        assert image.max() <= 1
        if image.ndim == 2:  # binary
            image = skimage.color.gray2rgb(image)
        assert image.ndim == 3
        return image

    def _get_activations(self, images, layer_names):
        layer_tensors = OrderedDict((layer, self.endpoints[layer]) for layer in layer_names)
        layer_outputs = self._sess.run(layer_tensors, feed_dict={self.inputs: images})
        return layer_outputs
