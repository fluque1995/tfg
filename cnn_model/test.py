#encoding: utf-8

from __future__ import print_function, division

import os
import sys
sys.path.append('..')

# Select CUDA device. Comment this line if working on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Import needed libraries
import keras
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import sco
from data_reader import DataReader
from models import ModelsDispatcher

# Select fold used as test set
curr_fold = 0

# Directory of saved models
save_dir = os.path.join(os.getcwd(), 'saved_models')

# Model name
model_name = 'model_cv{}.h5'.format(curr_fold)

# Number of classes for prediction
num_classes = 4

# Data reading from memory
reader = DataReader('../dataset')

# Test set loading
(x_test, y_test) = reader.load_dataset(curr_fold)[1]

# CNN model instantiation
dispatcher = ModelsDispatcher()
model = keras.models.load_model(
    os.path.join(save_dir, model_name),
    custom_objects={'sco': sco}
)

# Conversion of labels to categorical
# (class i -> [0,0,...,1,...,0] with 1 in i-th position)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# Model evaluation
scores = model.evaluate(x_test, y_test_cat, verbose=1)

# Labels prediction
y_pred = model.predict(x_test)

# Conversion from categorical to index
final_preds = [i for i in np.argmax(y_pred, axis=1)]

# Confusion matrix calculation
conf_mat = confusion_matrix(y_test, final_preds)

# Sum by rows and columns for score calculation
column_sums = np.sum(conf_mat, axis = 1)
row_sums = np.sum(conf_mat, axis = 0)

# Results printing
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Confusion matrix:')
print(conf_mat)

print("Partial results:")
t_n = 2*conf_mat[0,0] / (column_sums[0] + row_sums[0])
t_a = 2*conf_mat[1,1] / (column_sums[1] + row_sums[1])
t_o = 2*conf_mat[2,2] / (column_sums[2] + row_sums[2])
t_noise = 2*conf_mat[3,3] / (column_sums[3] + row_sums[3])

print("    Normal ECGs score: {}".format(t_n))
print("    AF ECGs score: {}".format(t_a))
print("    Other ECGs score: {}".format(t_o))
print("    Noisy ECGs score: {}".format(t_noise))

print("Total score: {}".format((t_n + t_a + t_o + t_noise)/4))
