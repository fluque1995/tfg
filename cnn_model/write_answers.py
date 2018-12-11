#encoding: utf-8

from __future__ import print_function, division

import os
import sys
import csv
sys.path.append('..')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import keras
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import sco
from data_reader import DataReader
from models import ModelsDispatcher

curr_fold = 4

save_dir = os.path.join(os.getcwd(), 'saved_models_final')
model_name = 'model_cv{}.h5'.format(curr_fold)
num_classes = 4

# Data CV generation
reader = DataReader('../dataset')

(x_test, y_test) = reader.load_dataset(curr_fold)[1]

dispatcher = ModelsDispatcher()
model = keras.models.load_model(
    os.path.join(save_dir, model_name),
    custom_objects={'sco': sco}
)

y_test_cat = keras.utils.to_categorical(y_test, num_classes)
scores = model.evaluate(x_test, y_test_cat, verbose=1)
y_pred = model.predict(x_test)

final_preds = [i for i in np.argmax(y_pred, axis=1)]

conf_mat = confusion_matrix(y_test, final_preds)
column_sums = np.sum(conf_mat, axis = 1)
row_sums = np.sum(conf_mat, axis = 0)

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

classes_list = ['N', 'A', 'O', '~']

answers_file = open("answers-{}.txt".format(curr_fold+1), "w")

writer = csv.writer(answers_file, delimiter=",")
f_reader = csv.reader(open('../dataset/folds/REFERENCE-{}.csv'.format(curr_fold+1), "r"))
i = 0

for row in f_reader:
    writer.writerow([row[0], classes_list[final_preds[i]]])
    i += 1
