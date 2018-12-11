from __future__ import division
from numpy import random
from keras import utils

import os
import csv
import wfdb
import numpy as np
import math

# Class DataReader
class DataReader:

    # Initialization
    def __init__(self, directory_name, fold_num=5):
        # Name of data directory
        self.directory_name = directory_name
        self.data_dir = os.path.join(directory_name, "data")
        self.folds_dir = os.path.join(directory_name, "folds")

        # Number of folds used for cross validation
        self.num_folds = fold_num

        # Numerical correspondence between labels and indices
        self.labels_correspondence = {
            'N': 0,
            'A': 1,
            'O': 2,
            '~': 3
        }

        (self.folds, self.labels) = self._load_data()

    # Data length unification. Zeroes are append after the
    # ECG in order to make them all equally long
    def _unify_length(self, data, max_length):
        for i in range(len(data)):
            data[i] =  np.concatenate(
                (
                    data[i],
                    np.zeros((max_length - data[i].shape[0], 1))
                ),
                axis = 0
            )
        return data

    # Data load using CSV specifications
    def _read_signals_from_csv(self, csv_reader, dir):
        x_set = []
        y_set = []
        max_length = 0
        for row in csv_reader:
            curr_signal = wfdb.rdrecord(os.path.join(dir,row[0])).__dict__['p_signal']
            curr_signal = curr_signal.reshape(-1, 1)
            curr_signal = (curr_signal - np.mean(curr_signal))/np.std(curr_signal)
            curr_length = curr_signal.shape[0]
            if curr_length > max_length:
                max_length = curr_length
            x_set.append(curr_signal)
            y_set.append(self.labels_correspondence[row[1]])

        return [x_set, y_set, max_length]

    # Full data load
    def _load_data(self):

        files = [open(os.path.join(self.folds_dir, 'REFERENCE-{}.csv'.format(i+1)))
                 for i in range(self.num_folds)]

        readers = [csv.reader(f) for f in files]

        folds = []
        labels = []
        max_length = 0

        for reader in readers:
            curr_fold, curr_labels, curr_length = self._read_signals_from_csv(
                reader,
                self.data_dir
            )

            folds.append(curr_fold)
            labels.append(curr_labels)

            if curr_length > max_length:
                max_length = curr_length

        unified_folds = [self._unify_length(fold, max_length) for fold in folds]
        return (folds, labels)

    # Building of test and training set depending on test fold index
    def load_dataset(self, test_fold):
        x_test = self.folds[test_fold]
        y_test = self.labels[test_fold]

        for i in range(self.num_folds):
            if i != test_fold:
                x_train = np.vstack(
                    (self.folds[i] for i in range(self.num_folds)
                     if i != test_fold)
                )
                y_train = np.hstack(
                    (self.labels[i] for i in range(self.num_folds)
                     if i != test_fold)
                )

        return (
            np.asarray(x_train),
            np.asarray(y_train)
        ), (
            np.asarray(x_test),
            np.asarray(y_test)
        )

    # Method that allows the split of data in validation sets
    def create_cross_validation_sets(self, n_sets=5):
        f = open(self.wd + self.fname_train, 'r')
        records_dict = self._create_dict_from_csv(f)
        splitted_dict = self._split_in_classes(records_dict)
        folds_dict = [{} for i in range(n_sets)]
        max_length = 0

        for key, value in splitted_dict.items():
            random.shuffle(value)
            size = len(value)
            i = 0
            for j in range(size):
                folds_dict[i][value[j]] = key
                i += 1
                i %= n_sets

        folds = [[] for i in range(n_sets)]
        labels = [[] for i in range(n_sets)]

        for i in range(n_sets):
            curr_fold_dict = folds_dict[i]
            for key in sorted(curr_fold_dict):
                curr_signal = wfdb.rdrecord(self.wd + key).__dict__['p_signal']
                curr_signal = curr_signal.reshape(1, -1)
                curr_length = curr_signal.shape[1]
                if curr_length > max_length:
                    max_length = curr_length
                folds[i].append(curr_signal)
                labels[i].append(self.labels_correspondence[curr_fold_dict[key]])

        for i in range(n_sets):
            folds[i] = self._process_raw_data(folds[i], max_length)

        return (folds, labels)


# Data generator
class DataGenerator(utils.Sequence):

    # Initialization
    def __init__(self, x_set, y_set, batch_size=128,
                 num_classes=4, subset='train', distortion_fact=0.1,
                 distortion_prob=0.5):
        self.x = x_set
        self.y = y_set
        self.subset = subset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dist_fact = distortion_fact
        self.dist_prob = 1-distortion_prob

        if subset == 'train':
            self.sep_dict = {key: [] for key in np.unique(self.y)}
            for i in range(len(self.x)):
                self.sep_dict[self.y[i]].append(self.x[i])
        elif subset != 'validation':
            raise ValueError("Unsupported subset type")

    # Return the number of different batches created by the generator
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    # Needed for the class to work properly
    def on_epoch_end(self):
        pass

    # Returns shape of a single data example
    def input_shape(self):
        return self.x[0].shape

    # Signal amplification by multiplying signal with random noise
    def amp_signal(self, signal):
        return random.uniform(1-self.dist_fact, 1+self.dist_fact)*signal

    # Signal vertical displacement by adding a fixed random amount
    def vertical_shift(self, signal):
        return random.uniform(-self.dist_fact, self.dist_fact) + signal

    # Horizontal shift by rolling the data
    def horizontal_shift(self, signal):
        shift = random.randint(len(signal))
        return np.roll(signal, shift)

    # Signal amplification for peaks
    def peaks_noise(self, signal):
        max_val = signal.max()
        max_threshold = .9*max_val
        affected_values = signal > max_threshold
        signal[affected_values] *= np.random.uniform(1-self.dist_fact, 1+self.dist_fact)
        return signal

    # Return of batches for training or validation set
    def __getitem__(self, idx):

        if self.subset == 'train':
            batch_x = []
            batch_y = []
            j = 0
            for i in range(self.batch_size):
                curr_index = random.randint(len(self.sep_dict[j]))
                curr_signal = self.sep_dict[j][curr_index]

                if random.uniform() > self.dist_prob:
                    curr_signal = self.amp_signal(curr_signal)

                if random.uniform() > self.dist_prob:
                    curr_signal = self.vertical_shift(curr_signal)

                if random.uniform() > self.dist_prob:
                    curr_signal = self.horizontal_shift(curr_signal)

                if random.uniform() > self.dist_prob:
                    curr_signal = self.peaks_noise(curr_signal)

                batch_x.append(curr_signal)
                batch_y.append(j)
                j += 1
                j %= self.num_classes

            return (
                np.asarray(batch_x),
                utils.to_categorical(batch_y, num_classes=self.num_classes)
            )

        elif self.subset == 'validation':
            batch_x = self.x[idx*self.batch_size : (idx + 1)*self.batch_size]
            batch_y = self.y[idx*self.batch_size : (idx + 1)*self.batch_size]

            batch_x_list = []
            for elem in batch_x:
                batch_x_list.append(elem)

            return (
                np.asarray(batch_x_list),
                utils.to_categorical(batch_y, num_classes=self.num_classes)
            )

    # Return the whole dataset
    def return_set(self):
        x_set = []
        for elem in self.x:
            x_set.append(elem)

        return (
            np.asarray(x_set),
            utils.to_categorical(self.y, num_classes=self.num_classes)
        )
