import logging
from collections import OrderedDict

import numpy as np
from brainio_base.assemblies import walk_coords, array_is_element, BehavioralAssembly
from numpy.random.mtrand import RandomState
from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.transformations import CrossValidationSingle
from brainscore.utils import fullname
from neural_nlp.analyze.data.ceiling import average_subregions
from neural_nlp.neural_data.fmri import load_voxels
from result_caching import store

accuracy = Accuracy()


def fit_decode(train, test):
    train, test = train.transpose('presentation', 'neuroid'), test.transpose('presentation', 'neuroid')
    decoder = TFProbabilitiesClassifier()
    decoder.fit(train, train['story'])
    train_acc = evaluate(decoder, train)
    test_acc = evaluate(decoder, test)
    train_acc, test_acc = train_acc.expand_dims('train_test'), test_acc.expand_dims('train_test')
    train_acc['train_test'], test_acc['train_test'] = ['train'], ['test']
    acc = Score.merge(train_acc, test_acc)
    return acc


def evaluate(decoder, assembly):
    probabilities = decoder.predict_proba(assembly)
    prediction = [probabilities['choice'].values[row.argmax()] for row in probabilities.values]
    acc = accuracy(prediction, assembly['story'].values)
    acc = acc.rename({'aggregation': 'accuracy_aggregation'})
    return acc


@store()
def decode_fROI():
    assembly = load_voxels()
    assembly = average_subregions(assembly)
    cross_validation = CrossValidationSingle(splits=10, split_coord='stimulus_id', stratification_coord='story')
    score = cross_validation(assembly, apply=fit_decode)
    return score


@store()
def decode_voxels():
    assembly = load_voxels()
    cross_validation = CrossValidationSingle(splits=1, split_coord='stimulus_id', stratification_coord='story')

    subjects = list(sorted(set(assembly['subject_UID'].values)))
    scores = []
    for subject in subjects:
        subject_index = [subject == coord_subject for coord_subject in assembly['subject_UID'].values]
        subject_assembly = assembly[{'neuroid': subject_index}]
        subject_score = cross_validation(subject_assembly, apply=fit_decode)
        subject_score = subject_score.sel(aggregation='center')  # since we're only doing one split
        print(f"subject {subject}: "
              f"{subject_score.sel(accuracy_aggregation='center', train_test='train').values:.2f} train, "
              f"{subject_score.sel(accuracy_aggregation='center', train_test='test').values:.2f} test")
        subject_score = subject_score.expand_dims('subject')
        subject_score['subject'] = [subject]
        scores.append(subject_score)
    scores = Score.merge(*scores)
    return scores


class TFProbabilitiesClassifier:
    def __init__(self,
                 init_lr=1e-4,
                 max_epochs=40,
                 zscore_feats=True,
                 train_batch_size=64,
                 eval_batch_size=240,
                 activation=None,
                 fc_weight_decay=0.463,  # 1/(C_svc * num_objectome_imgs) = 1/(1e-3 * 2160),
                 # based on https://stats.stackexchange.com/questions/216095/how-does-alpha-relate-to-c-in-scikit-learns-sgdclassifier
                 fc_dropout=1.0,
                 tol=1e-4,
                 log_rate=5, gpu_options=None):
        """
        mapping function class.
        :param train_batch_size: train batch size
        :param eval_batch_size: prediction batch size
        :param activation: what activation to use if any
        :param init_lr: initial learning rate
        :param zscore_feats: whether to zscore model features
        :param tol: tolerance - stops the optimization if reaches below tol
        :param fc_weight_decay: regularization coefficient for fully connected layers (inverse of sklearn C)
        :params fc_dropout: dropout parameter for fc layers
        :param log_rate: rate of logging the loss values (in epochs)
        """
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._zscore_feats = zscore_feats
        self._lr = init_lr
        self._tol = tol
        self._activation = activation
        self._fc_weight_decay = fc_weight_decay
        self._fc_dropout = fc_dropout
        self._max_epochs = max_epochs
        self._log_rate = log_rate
        self._gpu_options = gpu_options

        self._graph = None
        self._lr_ph = None
        self._opt = None
        self._scaler = None
        self._logger = logging.getLogger(fullname(self))

    def _iterate_minibatches(self, inputs, targets=None, batchsize=240, shuffle=False, random_state=None):
        """
        Iterates over inputs with minibatches
        :param inputs: input dataset, first dimension should be examples
        :param targets: [n_examples, ...] response values, first dimension should be examples
        :param batchsize: batch size
        :param shuffle: flag indicating whether to shuffle the data while making minibatches
        :return: minibatch of (X, Y)
        """
        input_len = inputs.shape[0]
        if shuffle:
            indices = np.arange(input_len)
            random_state.shuffle(indices)
        for start_idx in range(0, input_len, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if targets is None:
                yield inputs[excerpt]
            else:
                yield inputs[excerpt], targets[excerpt]

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

    def initializer(self, kind='xavier', *args, **kwargs):
        import tensorflow as tf
        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(*args, **kwargs, seed=0)
        else:
            init = getattr(tf, kind + '_initializer')(*args, **kwargs)
        return init

    def fc(self,
           inp,
           out_depth,
           kernel_init='xavier',
           kernel_init_kwargs=None,
           bias=1,
           weight_decay=None,
           activation=None,
           name='fc'):

        import tensorflow as tf
        if weight_decay is None:
            weight_decay = 0.
        # assert out_shape is not None
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}
        resh = inp
        assert (len(resh.get_shape().as_list()) == 2)
        in_depth = resh.get_shape().as_list()[-1]

        # weights
        init = self.initializer(kernel_init, **kernel_init_kwargs)
        kernel = tf.get_variable(initializer=init,
                                 shape=[in_depth, out_depth],
                                 dtype=tf.float32,
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 name='weights')
        init = self.initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                 shape=[out_depth],
                                 dtype=tf.float32,
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 name='bias')

        # ops
        fcm = tf.matmul(resh, kernel)
        output = tf.nn.bias_add(fcm, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        return output

    def _make_behavioral_map(self):
        """
        Makes the temporal mapping function computational graph
        """
        import tensorflow as tf
        num_classes = len(self._label_mapping.keys())
        with self._graph.as_default():
            with tf.variable_scope('behavioral_mapping'):
                out = self._input_placeholder
                out = tf.nn.dropout(out, keep_prob=self._fc_keep_prob, name="dropout_out")
                pred = self.fc(out,
                               out_depth=num_classes,
                               activation=self._activation,
                               weight_decay=self._fc_weight_decay, name="out")

                self._predictions = pred

    def _make_loss(self):
        """
        Makes the loss computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.variable_scope('loss'):
                logits = self._predictions

                self.classification_error = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._target_placeholder))
                self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                self.total_loss = self.classification_error + self.reg_loss
                self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                                   global_step=tf.train.get_or_create_global_step())

    def _init_mapper(self, X, Y):
        """
        Initializes the mapping function graph
        :param X: input data
        """
        import tensorflow as tf
        assert len(Y.shape) == 1
        with self._graph.as_default():
            self._input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
            self._target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
            self._fc_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
            # Build the model graph
            self._make_behavioral_map()
            self._make_loss()

            # initialize graph
            self._logger.debug('Initializing mapper')
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess = tf.Session(
                config=tf.ConfigProto(gpu_options=self._gpu_options) if self._gpu_options is not None else None)
            self._sess.run(init_op)

    def labels_to_indices(self, labels):
        label2index = OrderedDict()
        indices = []
        for label in labels:
            if label not in label2index:
                label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
            indices.append(label2index[label])
        index2label = OrderedDict((index, label) for label, index in label2index.items())
        return np.array(indices), index2label

    def fit(self, X, Y):
        """
        Fits the parameters to the data
        :param X: Source data, first dimension is examples
        :param Y: Target data, first dimension is examples
        """
        if self._zscore_feats:
            import sklearn
            self._scaler = sklearn.preprocessing.StandardScaler().fit(X)
            X = self._scaler.transform(X)
        Y, self._label_mapping = self.labels_to_indices(Y.values)
        self.setup()
        assert X.ndim == 2, 'Input matrix rank should be 2.'
        random_state = RandomState(0)
        with self._graph.as_default():
            self._init_mapper(X, Y)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc='epochs'):
                for counter, batch in enumerate(
                        self._iterate_minibatches(X, Y, batchsize=self._train_batch_size, shuffle=True,
                                                  random_state=random_state)):
                    feed_dict = {self._input_placeholder: batch[0],
                                 self._target_placeholder: batch[1],
                                 self._lr_ph: lr,
                                 self._fc_keep_prob: self._fc_dropout}
                    _, loss_value, reg_loss_value = self._sess.run(
                        [self.train_op, self.classification_error, self.reg_loss],
                        feed_dict=feed_dict)
                if epoch % self._log_rate == 0:
                    self._logger.debug(f'Epoch: {epoch}, Err Loss: {loss_value:.2f}, Reg Loss: {reg_loss_value:.2f}')

                if loss_value < self._tol:
                    self._logger.debug('Converged.')
                    break

    def predict_proba(self, X):
        import tensorflow as tf
        assert len(X.shape) == 2, "expected 2-dimensional input"
        if self._zscore_feats:
            scaled_X = self._scaler.transform(X)
        else:
            scaled_X = X
        with self._graph.as_default():
            preds = []
            for batch in self._iterate_minibatches(scaled_X, batchsize=self._eval_batch_size, shuffle=False):
                feed_dict = {self._input_placeholder: batch, self._fc_keep_prob: 1.0}
                preds.append(np.squeeze(self._sess.run([tf.nn.softmax(self._predictions)], feed_dict=feed_dict)))
            proba = np.concatenate(preds, axis=0)
        # we take only the 0th dimension because the 1st dimension is just the features
        X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                    if array_is_element(dims, X.dims[0])}
        proba = BehavioralAssembly(proba,
                                   coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                   dims=[X.dims[0], 'choice'])
        return proba

    def close(self):
        """
        Closes occupied resources
        """
        import tensorflow as tf
        tf.reset_default_graph()
        self._sess.close()


if __name__ == '__main__':
    score = decode_voxels()  # ~20% train, ~5% test
    print(score)
    score = decode_fROI()  # about 12% train & test
    print(score)

