"""Abstract estimator module, will contain just 1 class."""

import os
import logging
import warnings
import time

import numpy as np
from sklearn.utils import shuffle
from sklearn.externals import joblib

from .. import inputs

class Classifier(object):
    """Abstract estimator class"""

    PERSIST_FILENAME = 'classifier.pkl'

    OK = 0
    GENERAL_ERROR = 1
    NO_DATASET = 2
    EVALUATE_LOW_SCORE = 4
    EVALUATE_NOT_ENOUGH_DATA = 8

    def __init__(self, modelid, directory):

        self.classes = None

        self.modelid = modelid

        self.runid = str(int(time.time()))

        self.persistencedir = os.path.join(directory, 'classifier')
        if os.path.isdir(self.persistencedir) is False:
            if os.makedirs(self.persistencedir) is False:
                raise OSError('Directory ' + self.persistencedir + ' can not be created.')

        # We define logsdir even though we may not use it.
        self.logsdir = os.path.join(directory, 'logs', self.get_runid())
        if os.path.isdir(self.logsdir):
            raise OSError('Directory ' + self.logsdir + ' already exists.')
        if os.makedirs(self.logsdir) is False:
            raise OSError('Directory ' + self.logsdir + ' can not be created.')

        # Logging.
        logfile = os.path.join(self.logsdir, 'info.log')
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        warnings.showwarning = self.warnings_to_log

        self.X = None
        self.y = None

        self.reset_metrics()

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=5)
        np.set_printoptions(threshold=np.inf)
        np.seterr(all='raise')


    @staticmethod
    def warnings_to_log(message, category, filename, lineno):
        """showwarnings overwritten"""
        logging.warning('%s:%s: %s:%s', filename, lineno, category.__name__, message)


    def get_runid(self):
        """Returns the run id"""
        return self.runid


    def load_classifier(self):
        """Loads a previously stored classifier"""
        classifier_filepath = os.path.join(self.persistencedir, Classifier.PERSIST_FILENAME)
        return joblib.load(classifier_filepath)

    def store_classifier(self, trained_classifier):
        """Stores the provided classifier"""
        classifier_filepath = os.path.join(self.persistencedir, Classifier.PERSIST_FILENAME)
        joblib.dump(trained_classifier, classifier_filepath)

    @staticmethod
    def get_labelled_samples(filepath, test_data=0.0):
        """Extracts labelled samples from the provided data file"""
        data_provider = inputs.DataProvider(filepath, [0, 1], training=True, test_data=test_data)
        return data_provider

    @staticmethod
    def get_unlabelled_samples(filepath):
        """Extracts unlabelled samples from the provided data file"""

        data_provider = inputs.DataProvider(filepath, [0, 1], training=False)
        return data_provider

    def reset_metrics(self):
        """Resets the class metrics"""
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.phis = []
