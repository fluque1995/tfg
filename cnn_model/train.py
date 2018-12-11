#encoding: utf-8
from __future__ import print_function, division

import os
import sys
sys.path.append('..')

# Select CUDA device. Comment this line if working on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Import needed libraries
import keras
import numpy as np

from sklearn.metrics import confusion_matrix
from utils import sco
from data_reader import DataReader, DataGenerator
from models import ModelsDispatcher

# Training process parameters
batch_size = 128
num_classes = 4
test_size = 0.2
validation_size = 1/6
epochs = 500

# Select fold used as test set
curr_fold = 1

# Directory of saved models
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model_cv{}.h5'.format(curr_fold)

# If directory for saved models doesn't exist, create it
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

# Data reading from memory
reader = DataReader('../dataset')

# Test and train loading
(x_train, y_train) = reader.load_dataset(curr_fold)[0]
(x_test, y_test) = reader.load_dataset(curr_fold)[1]

# Training data generator instantiation
train_generator = DataGenerator(
    x_train,
    y_train,
    batch_size=batch_size
)

# Test data generator instantiation
test_generator = DataGenerator(x_test, y_test, subset='validation')

# CNN model instantiation
dispatcher = ModelsDispatcher()
model = dispatcher.basic_model(
    input_shape = train_generator.input_shape(),
    num_classes = num_classes
)

# Initialization of RMSprop optimizer
opt = keras.optimizers.RMSprop()

# Initialization of checkpointer
checkpointer = keras.callbacks.ModelCheckpoint(
    model_path,
    verbose=1,
    monitor='val_acc',
    save_best_only=True,
    mode='max',
    period=10
)

# Model summary printing
print(model.summary())

# Conversion of labels to categorical
# (class i -> [0,0,...,1,...,0] with 1 in i-th position)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', sco])

# Model training process
model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=15,
    callbacks=[checkpointer],
)

# After training process, evaluate the trained model
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
