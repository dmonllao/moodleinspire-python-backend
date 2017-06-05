"""Tensorflow classifier"""

from __future__ import division

import math
import os

from sklearn import preprocessing
import tensorflow as tf
import numpy as np

class TF(object):
    """Tensorflow classifier"""

    N_EPOCH = 100

    def __init__(self, data_provider, starter_learning_rate, tensor_logdir):

        self.starter_learning_rate = starter_learning_rate
        self.n_features = data_provider.get_features_number()
        self.classes = data_provider.get_classes()
        self.tensor_logdir = tensor_logdir

        self.fit_index = 0

        self.x = None
        self.y_ = None
        self.y = None
        self.z = None
        self.loss = None

        self.build_graph()

        self.start_session()

        # During evaluation we process the same dataset multiple times, could could store
        # each run result to the user but results would be very similar we only do it once
        # making it simplier to understand and saving disk space.
        if os.listdir(self.tensor_logdir) == []:
            self.log_run = True
            self.init_logging()
        else:
            self.log_run = False

    def  __getstate__(self):
        state = self.__dict__.copy()
        del state['x']
        del state['fit_index']
        del state['y_']
        del state['y']
        del state['z']
        del state['train_step']
        del state['sess']

        del state['file_writer']
        del state['merged']

        # We also remove this as it depends on the run.
        del state['tensor_logdir']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.build_graph()
        self.start_session()

    def set_tensor_logdir(self, tensor_logdir):
        '''Needs to be set separately as it depends on the run, it can not be restored.'''
        self.tensor_logdir = tensor_logdir
        try:
            self.file_writer
            self.merged
        except AttributeError:
            # Init logging if logging vars are not defined.
            self.init_logging()

    def build_graph(self):
        """Builds the computational graph without feeding any data in"""

        # Placeholders for input values.
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float64, [None, self.n_features], name='x')
            self.y_ = tf.placeholder(tf.float64, [None, len(self.classes)], name='dataset-y')

        # Variables for computed stuff, we need to initialise them now.
        with tf.name_scope('weights'):
            W = tf.Variable(tf.zeros([self.n_features, len(self.classes)], dtype=tf.float64),
                            name='weights')
            b = tf.Variable(tf.zeros([len(self.classes)], dtype=tf.float64), name='bias')

        # Predicted y.
        with tf.name_scope('activation'):
            self.z = tf.matmul(self.x, W) + b
            tf.summary.histogram('predicted_values', self.z)
            self.y = tf.nn.softmax(self.z)
            tf.summary.histogram('activations', self.y)

        with tf.name_scope('loss'):
            cross_entropy = - tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.y, -1.0, 1.0)))
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Calculate decay_rate.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                   100, 0.96, staircase=False)
        tf.summary.scalar("learning_rate", learning_rate)

        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    def start_session(self):
        """Starts the session"""

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    def init_logging(self):
        """Starts logging the tensors state"""
        self.file_writer = tf.summary.FileWriter(self.tensor_logdir, self.sess.graph)
        self.merged = tf.summary.merge_all()

    def get_session(self):
        """Return the session"""
        return self.sess

    def fit(self, data_provider):
        """Fits provided data into the session"""

        self.fit_index = 0
        it = data_provider.get_data()
        for x, y in it:

            for _ in range(self.N_EPOCH):

                # Check that the batch contains all different classes samples.
                if len(np.unique(y)) < len(self.classes):
                    # Convert to a proper error message.
                    print('Unbalanced classes');
                    continue

                # Switch from a samples classes vector to a multi-class matrix.
                # TODO Move it to preprocessing using self.classes ONLY FOR TRAINING DATA, NOT TEST DATA.
                y_multi = preprocessing.MultiLabelBinarizer().fit_transform(y.reshape(len(y), 1))

                if self.log_run:
                    _, summary = self.sess.run([self.train_step, self.merged],
                                               {self.x: x, self.y_: y_multi})
                    # Add the summary data to the file writer.
                    self.file_writer.add_summary(summary, self.fit_index)
                else:
                    self.sess.run(self.train_step, {self.x: x, self.y_: y_multi})

                # Bump fit count so summary indexes are increased.
                self.fit_index = self.fit_index + 1

    def predict(self, data_provider):
        """Returns predictions with the prediction probabilities, all related to samples."""

        y_proba = []
        y_pred = []
        sampleids = []

        it = data_provider.get_data()
        for samples, x in it:
            sampleids = sampleids + samples.tolist()
            y_proba = y_proba + self.predict_proba(x).tolist()
            y_pred = y_pred + self.predict_label(x).tolist()

        # Probabilities of the predicted response being correct.
        y_proba = np.array(y_proba)
        probabilities = y_proba[range(len(y_proba)), y_pred]

        # First column sampleids, second the prediction and third how
        # reliable is the prediction (from 0 to 1).
        return np.vstack((sampleids, y_pred, probabilities)).T.tolist()

    def test(self, data_provider):

        y_test = []
        y_proba = []
        y_pred = []

        it = data_provider.get_test_data()
        for x, y in it:
            y_test = y_test + y.tolist()
            y_proba = y_proba + self.predict_proba(x).tolist()
            y_pred = y_pred + self.predict_label(x).tolist()

        y_proba = np.array(y_proba)
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        y_score = y_proba[range(len(y_proba)), y_test]

        return [y_test, y_score, y_pred]

    def predict_label(self, x):
        """Returns predictions"""
        return self.sess.run(tf.argmax(self.y, 1), {self.x: x})

    def predict_proba(self, x):
        """Returns predicted probabilities"""
        return self.sess.run(self.z, {self.x: x})
