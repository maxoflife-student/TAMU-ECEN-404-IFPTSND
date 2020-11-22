import sys

from os import listdir
from os.path import isfile, join

import tensorflow as tf

import pandas as pd
import numpy as np

from IPython.display import display

import ipywidgets as widgets
from ipywidgets import Layout
from bqplot import (
    LinearScale, Lines, Axis, Figure, Toolbar, ColorScale
)

import json

import os
import datetime
import time
import sys

from pathlib import Path

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import *

from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow import math

import tensorflow.keras.backend as kb
from tensorflow.python.ops import math_ops

from explore_entities import Graph_Entities

from tensorflow.keras.layers import Dense, Dropout, LSTM, Multiply


# By specifying the alpha value of the first function, we can return a rank_loss_function with a certain value
# for alpha that can be called in any instance
def rank_loss_func(alpha=1):
    def rlf(y_actual, y_pred):
        # Slice the final predicted time step from the actual value
        y_slice = y_actual[:, -1:]
        # Calculate return ratio
        return_ratio = tf.math.divide(tf.math.subtract(y_pred, y_slice), y_slice)
        # Create an array of all_ones so that we can calculate all permutations of subtractions
        all_ones = tf.ones([y_slice.shape[0], 1])

        # Creates a N x N matrix with every predicted return ratio for each company subtracted with every other
        # company
        pred_dif = tf.math.subtract(
            tf.matmul(return_ratio, all_ones, transpose_b=True),
            tf.matmul(all_ones, return_ratio, transpose_b=True)
        )

        # Creates an N x N matrix containing every actual return ratio for each company subtracted with every other
        # company By switching the order of the all_ones matricies and the actual prices, a negative sign is introduced
        # When RELU is applied later, correct predictions will not affect loss while incorrect predictions will affect
        # loss depending on how incorrect the prediction was
        actual_dif = tf.math.subtract(
            tf.matmul(all_ones, y_slice, transpose_b=True),
            tf.matmul(y_slice, all_ones, transpose_b=True)
        )

        # Using the above two qualities, the algorithm can be punished for incorrectly calculating when a company is
        # doing better than another company Reduces the mean across each dimension until only 1 value remains
        rank_loss = tf.reduce_mean(
            # Takes if a given value is >0, it is kept, otherwise, it becomes 0
            tf.nn.relu(
                # Multiplies all of the
                tf.multiply(pred_dif, actual_dif)
            )
        )

        # Take the squared difference between the y_actual and y_pred
        squared_difference = math_ops.squared_difference(y_actual, y_pred)
        # Average each of values, taking the magnitude
        sd_mean = kb.mean(squared_difference, axis=-1)

        loss = tf.cast(alpha, tf.float32) * rank_loss + sd_mean
        return loss

    # Return the RLF function instance
    return rlf


# Define a func using the Tensorflow 2.0 implementation of Leaky ReLU
leaky_relu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.2)(x)


# Define a custom layer in Tensorflow 2.0 to implement a matrix multiplication
# operation using the output of an LSTM layer and another given matrix
class Ein_Multiply(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(Ein_Multiply, self).__init__()

    def call(self, input):
        return tf.einsum('ij,ik->ik', input[0], input[1])


class TF_Models(Graph_Entities):
    def __init__(self, data_path, models_path):
        self.data_path = data_path
        self.models_path = models_path

        # Re-use the methods from explore_entities
        self.entities, self.entities_idx = super()._generate_list_of_entities(data_path)
        # self.relations_dict = super()._generate_relations()
        # self.Normalized_Adjacency_Matrix = super()._generate_normalized_ajacency_matrix()

        self.XX_tf = self._load_data_into_TF()
        self.YY_tf = self._create_labels()

    '''Returns a 3 Dimensional Tensor with dimensions: (Entities, Sequences, Features)'''
    '''For the LSTM model, the first dimension is a placeholder, so entities must be first'''

    def _load_data_into_TF(self):
        XX_t = []
        for ent in self.entities:
            df = pd.read_csv(self.data_path + '/' + ent + '.csv')
            # Append the 2x2 tensor of each company's features through all time-sequences to the main tensor
            XX_t.append(df[0:].values)

        XX_t = tf.constant(XX_t)
        return XX_t

    '''This function should be modified depending on the specific use case. In IFPTSND, the labels should be
        the return ratio of day t+1. However, that is project specific and should be changed if financial market
        predictions are the final labels'''

    def _create_labels(self):

        XX_t = self.XX_tf.numpy()

        # Labels are currently the same day close price
        YY_t = XX_t[:, :, 0]
        # Labels are now the next day close price
        YY_t = YY_t[:, 1:]
        # Now that the number of time steps don't match between XX and YY,
        # the last day of XX_tf needs to be removed since we don't have a prediction for it
        self.XX_tf = tf.constant(XX_t[:, 0:-1, :])

        # In our project, the labels shouldn't be the actual price of the next day. It needs to be the return ratio
        # had we purchased the stock
        YY_tf = np.copy(YY_t)
        for e in range(XX_t.shape[0]):
            for t in range(XX_t.shape[1] - 1):
                YY_tf[e, t] = (YY_t[e, t] - XX_t[e, t, 0]) / XX_t[e, t, 0]

                # Some days the price won't change between days
                # Those zeroes need to be made really small, but not actually 0 to avoid loss errors later
                if YY_tf[e, t] == 0:
                    YY_tf[e, t] = 1e-10

        return tf.constant(YY_tf)

    '''Splits the data into a training, validation, and testing set. The proportion of entries in each category
        can be set using the list. For example by default, 60% training, 15% validation, 25% testing. Axis specifies
        which dimension to make the split. In the default example, it's time'''
    '''returns a dictionary containing the splits'''

    def split_data(self, axis=1, perc_split=[60, 15, 25]):

        # Given a total and list of splits, evenly distributes the total amount proportional to the given list
        def split_windows(total, percentages_list):
            # Get a sum of the initial total
            percentage_sum = sum(percentages_list)

            # Create a new list based on a percentage of the total
            new_splits = []
            for perc in percentages_list:
                new_splits.append(int(total * (perc / percentage_sum)))

            if sum(new_splits) != total:
                new_splits[0] = new_splits[0] + (total - sum(new_splits))

            return new_splits

        x_train, x_val, x_test = tf.split(self.XX_tf, split_windows(self.XX_tf.shape[1], perc_split),
                                          axis=axis)
        y_train, y_val, y_test = tf.split(self.YY_tf, split_windows(self.YY_tf.shape[1], perc_split),
                                          axis=axis)

        return {'x_train': x_train, 'y_train': y_train,
                'x_val': x_val, 'y_val': y_val,
                'x_test': x_test, 'y_test': y_test}

    def generate_model(self, input_shape, model_type='lstm', gcn_shape=None, loss_function='mse', activation='relu', learning_rate=1e-5, decay_rate=1e-6, hidden_units=64):

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate)

        '''Types of model inputs'''
        # Sequential Input Data
        input_seq = keras.Input(shape=input_shape)
        # Normalized Ajacency Matrix
        input_rel = keras.Input(shape=gcn_shape)

        '''Keras Layers'''
        LSTM_t = LSTM(hidden_units, return_sequences=True, activation=activation)
        LSTM_f = LSTM(hidden_units, return_sequences=False, activation=activation)
        Dense_u = Dense(64, activation=activation)
        Dense_o = Dense(1, activation=activation)

        '''Custom Layers'''
        # Ein_Multiply

        # One LSTM layer with return sequences, One LSTM without return sequences, One Dense Layer: (None, 1)
        if model_type == 'lstm':
            x = LSTM_t(input_seq)
            x = LSTM_f(x)
            x = Dense_o(x)
            model = tf.keras.Model(inputs=[input_seq], outputs=x)

        # Two LSTM layers, One GCN Layer, One Dense Layer: (None, 1)
        if model_type == 'lstm_gcn_1':
            x = LSTM_t(input_seq)
            x = LSTM_f(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense_u(x)
            x = Dense_o(x)
            model = tf.keras.Model(inputs=[input_seq], outputs=x)
            None

        # Two LSTM layers, Two GCN Layers, One Dense Layer: (None, 1)
        if model_type == 'lstm_gcn_2':
            x = LSTM_t(input_seq)
            x = LSTM_f(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense_u(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense_u(x)
            x = Dense_o(x)
            model = tf.keras.Model(inputs=[input_seq], outputs=x)
            None

        else:
            None

        model.compile(loss=loss_function, optimizer=optimizer)
        model.summary()

        return model


A = TF_Models('./data_sets/NASDAQ_Cleaned', './models')
A.generate_model(A.XX_tf.shape[1:])

