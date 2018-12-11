from __future__ import division
import numpy as np
import sklearn.metrics as mt
import tensorflow as tf
import keras.backend as K

def sco(y_true, y_pred):
    y_pred_classes = K.argmax(y_pred, axis=-1)
    y_true_classes = K.argmax(y_true, axis=-1)
    conf_mat = tf.confusion_matrix(
        y_true_classes,
        y_pred_classes,
        num_classes=4
    )

    column_sums = tf.reduce_sum(conf_mat, axis = 1)
    row_sums = tf.reduce_sum(conf_mat, axis = 0)

    t_n = tf.cast(2*conf_mat[0,0]/(column_sums[0] + row_sums[0]), tf.float32)
    t_a = tf.cast(2*conf_mat[1,1]/(column_sums[1] + row_sums[1]), tf.float32)
    t_o = tf.cast(2*conf_mat[2,2]/(column_sums[2] + row_sums[2]), tf.float32)
    t_noise = tf.cast(2*conf_mat[3,3]/(column_sums[3] + row_sums[3]), tf.float32)

    t_n = tf.where(tf.is_nan(t_n), tf.cast(0, tf.float32), t_n)
    t_a = tf.where(tf.is_nan(t_a), tf.cast(0, tf.float32), t_a)
    t_o = tf.where(tf.is_nan(t_o), tf.cast(0, tf.float32), t_o)
    t_noise = tf.where(tf.is_nan(t_noise), tf.cast(0, tf.float32), t_noise)

    return (t_n + t_a + t_o + t_noise) / 4
