import csv
import random
import warnings
from itertools import islice

import numpy as np
import tensorflow as tf

class DataProvider(object):

    def __init__(self, filepath, classes, training=True, test_data=0.0, batch_size=1000):

        self.training = training
        self.classes = classes
        self.set_features_number(filepath)

        if test_data <= 0.0:
            # Prepare dataset iterators.
            f = open(filepath, 'r')
            self.data_it = DataIterator(f, batch_size, self.get_features_number(), training=training)

            self.test_data_it = []
        else:

            # File lines.
            length = self.get_file_length(filepath)

            # Amount of test lines.
            test_length = int(round(length * test_data))

            # From 3 (after metadata & headers) to (len - test_length) so it is easier to calculate
            # where samples belong to.
            test_start_line = random.randint(3, length - test_length)
            test_stop_line = test_start_line + test_length

            # Limit the batch_size if test data is required.
            if batch_size > length:
                batch_size = test_length

            # Training dataset iterator.
            f = open(filepath, 'r')
            self.data_it = DataIterator(f, batch_size, self.get_features_number(),
                                        training=training, skip_start=test_start_line,
                                        skip_stop=test_stop_line)

            # Test dataset iterator.
            test_f = open(filepath, 'r')
            self.test_data_it = DataIterator(test_f, batch_size, self.get_features_number(),
                                             offset=test_start_line, stop=test_stop_line)

    def set_features_number(self, filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            metadata_vars = next(reader)
            metadata_values = next(reader)
            metadata = dict(zip(metadata_vars, metadata_values))
            self.n_features = int(metadata['nfeatures'])

    def get_file_length(self, filepath):
        with open(filepath, 'r') as f:
            for i, l in enumerate(f):
                pass

        # -3 because of file metadata and headers.
        return i + 1 - 3

    def get_classes(self):
        return self.classes

    def get_features_number(self):
        return self.n_features

    def get_data(self):
        return self.data_it

    def get_test_data(self):
        return self.test_data_it

class DataIterator(object):

    def __init__(self, file_iterator, batch_size, n_features, training=True, offset=0,
                 stop=None, skip_start=None, skip_stop=None):

        self.file_iterator = file_iterator
        self.batch_size = batch_size
        self.training = training
        self.n_features = n_features

        self.offset = offset
        self.stop = stop
        self.skip_start = skip_start
        self.skip_stop = skip_stop

        # Preset to the start.
        self.latest_line = 0

        if offset == 0:
            # Start equals 0 for training / prediction data, testing data starts, minimum at line 3.
            self.skipped_headers = False
        else:
            self.skipped_headers = True

    def __iter__(self):
        return self

    def next(self):

        if self.skipped_headers == True:
            skip_header = 0
        else:
            # Metadata on top of the file.
            skip_header = 3
            self.skipped_headers = True

        # Start index & batch length (possibly overwritten below).
        offset = 0
        n_lines = self.batch_size

        if self.offset:
            # Apply initial offset.
            offset = self.offset
            self.offset = None

        if self.skip_start:
            if self.latest_line + offset == self.skip_start:
                # Skip the lines that are supposed to be skipped.
                offset = offset + (self.skip_stop - self.skip_start)
            elif self.latest_line + offset + self.batch_size > self.skip_start and self.latest_line < self.skip_stop:
                # Limit the batch to self.skip_start.
                n_lines = self.skip_start - self.latest_line

        if self.stop:
            if self.latest_line + offset == self.stop:
                # We are done.
                raise StopIteration
            elif self.latest_line + offset + self.batch_size > self.stop and self.latest_line < self.stop:
                # Limit the batch to self.stop
                n_lines = self.stop - self.latest_line

        gen = islice(self.file_iterator, offset, n_lines + offset)

        # Read self.batch_size lines.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # Unfortunately getfromtxt generates a warning on StopIteration,
            # we don't want to show it

            if self.training:
                # All of them returned as float we later covert the label to int.
                dtypes = []
                for i in range(self.n_features):
                    dtypes.append(('f' + str(i), 'float32'))
                dtypes.append(('label', 'int'))
                data = np.genfromtxt(gen, delimiter=',', dtype=dtypes, skip_header=skip_header,
                                     missing_values='', filling_values=False)
            else:
                # All of them returned as float but the sampleid, which follows a x-y format
                # where both x and y are numeric values. 16 chars should be enough to cover
                # Moodle's 10 chars db scheme integers + the range index (usually just 1 char).
                dtypes = [('sampleid', 'S16')]
                for i in range(self.n_features):
                    dtypes.append(('f' + str(i), 'float32'))
                data = np.genfromtxt(gen, delimiter=',', dtype=dtypes, skip_header=skip_header,
                                     missing_values='', filling_values=False)

            # Update the latest line record.
            self.latest_line = self.latest_line + n_lines + offset

        # Convert to array if there is only 1 item as genfromtxt returns just a tuple.
        if n_lines == 1:
            data = np.array([data])

        if len(data) == 0:
            raise StopIteration

        column_names = list(data.dtype.names)
        if self.training:
            # np.delete(elem1, -1, axis=1) does not work...
            no_label = column_names[:-1]
            elem1 = map(list, data[no_label])
            elem2 = data['label']
        else:
            elem1 = data['sampleid']
            # np.delete(elem2, 0, axis=1) does not work...
            no_sampleid = column_names[1:]
            elem2 = map(list, data[no_sampleid])

        return elem1, elem2
