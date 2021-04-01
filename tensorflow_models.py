import datetime
import pickle
import sys
from os.path import join, isfile
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import *

from tensorflow import keras
import tensorflow.keras.backend as kb
from tensorflow.python.ops import math_ops

from tensorflow.keras.layers import Dense, LSTM

from explore_entities import Graph_Entities
import ipywidgets as widgets
from ipywidgets import Layout, GridBox
from IPython.display import display

import warnings
import os
import math

from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# By specifying the alpha value of the first function, we can return a rank_loss_function with a certain value
# for alpha that can be called in any instance
def rank_loss_func(rr_train, rr_val, alpha=1e-6, beta=1, forecast=1):
    if rr_train.shape == rr_val.shape:
        sys.exit("""This jank portion of code will cause an error because your validation and training set are
                 the same size""")

    def rlf(y_actual, y_pred):
      
      # Must be explicitly define given how None is the dimension size
      all_ones = None
      ground_truth = None
      if y_actual[0, 0] == 0:
        # all_ones = tf.ones([881, 309], dtype=tf.float32)
        forecast = 681-309
        ground_truth = tf.constant(rr_val, dtype=tf.float32)
      else:
        forecast = 0
        # all_ones = tf.ones([881, 681], dtype=tf.float32)
        ground_truth = tf.constant(rr_train, dtype=tf.float32)

      y_actual = y_actual[:, -1:]
      # y_pred = y_pred[:, forecast:]

      mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_actual, y_pred))

      # Calculate return ratio
      return_ratio = tf.math.divide(tf.math.subtract(y_pred, y_actual), y_actual)
      # return_ratio = y_pred

      # If the size of the calculated return-ratios does not match, it is known whether we are
      # in validation or in training (as long as validation and training are not the same size)
      
      # if y_actual[0, 0] == 0:
      #     ground_truth = tf.constant(rr_val, dtype=tf.float32)
      # else:
      #     ground_truth = tf.constant(rr_train, dtype=tf.float32)
      # ground_truth = y_actual

      # We only want the ground_truth for the last day being forecast
      # In the future maybe we can get the RL value for multi-timestep
      # For now we cannot
      ground_truth = ground_truth[:, -1:]
      # return_ratio = return_ratio[:, -1:]
      all_ones = tf.ones([881, 1], dtype=tf.float32)

      ###############################################################
      # Take the squared difference between the y_actual and y_pred
      # squared_difference = math_ops.squared_difference(ground_truth, return_ratio)
      # sd_mean = math_ops.reduce_mean(squared_difference)
      # mse = mean_squared_error(ground_truth.numpy(), return_ratio.numpy())

      # Create an array of all_ones so that we can calculate all permutations of subtractions
      # all_ones = tf.ones(y_actual.shape, dtype=tf.float32)

      # tf.print(f'{ground_truth.shape}', output_stream=sys.stdout)
      # tf.print(f'{return_ratio.shape}', output_stream=sys.stdout)
      # tf.print(f'{all_ones.shape}', output_stream=sys.stdout)

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
          tf.matmul(all_ones, ground_truth, transpose_b=True),
          tf.matmul(ground_truth, all_ones, transpose_b=True)
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

      # tf.print(tf.cast(alpha, tf.float32) * rank_loss, output_stream=sys.stdout)
      # tf.print(mse.shape, output_stream=sys.stdout)
      # tf.print(tf.multiply(pred_dif, actual_dif).shape, output_stream=sys.stdout)
      

      # Multiply the rank-loss term by alpha AND add the MSE to create total loss
      # loss = tf.cast(alpha, tf.float32) * rank_loss + sd_mean

      # kb.print_tensor(sd_mean)
      # kb.print_tensor(tf.cast(10000, tf.float32) * rank_loss)

      # Basically a check to see if the first item is padded,
      # when it is, we know we're in the validation set
      # if y_actual[0, 0] == 0:
      #   mse = tf.reduce_mean(mse) * 681 / 309
      # else:
      #   mse = tf.reduce_mean(mse)

      loss = (tf.cast(alpha, tf.float32) * rank_loss) + mse
      # tf.print("\n Rankloss ", output_stream=sys.stdout)
      # tf.print((tf.cast(alpha, tf.float32) * rank_loss), output_stream=sys.stdout)
      # tf.print("\n Rankloss ", output_stream=sys.stdout)

      return loss

    # Return the RLF function instance
    return rlf

# If the predicted labels are RR, then we must modify the function
def rank_loss_func_rr(rr_train, rr_val, alpha=1e-6, beta=1, forecast=1):

    if rr_train is not None:
        if rr_train.shape == rr_val.shape:
            sys.exit("""This jank portion of code will cause an error because your validation and training set are
                     the same size""")

    def rlf(y_actual, y_pred):

        # # Slice the final predicted time step from the actual value
        # # This represents how many days into the future we're forecasting
        # y_actual = y_actual[:, -forecast:]
        # y_pred = y_pred[:, -forecast:]

        # Calculate the MSE between each day's RR prediction and the actual
        mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_actual, y_pred))

        # Return the whole MSE as the loss
        loss = mse

        return loss

    # Return the RLF function instance
    return rlf


def mse_rr(y_actual, y_pred):
    # Slice the final predicted time step from the actual value
    # This represents how many days into the future we're forecasting
    y_actual_last = y_actual[:, -1:]
    y_pred_last = y_pred[:, -1:]

    # Calculate return ratio
    pred_return_ratio = tf.math.divide(tf.math.subtract(y_pred_last, y_actual_last), y_actual_last)
    actual_return_ratio = tf.math.divide(tf.math.subtract(y_actual[:, -2:-1], y_actual[:, -1:]), y_actual[:, -2:-1])

    mse = tf.keras.losses.mean_squared_error(pred_return_ratio, actual_return_ratio)

    loss = mse

    return loss

# These are the ones we've mainly used
def rank_loss_rr(y_actual, y_pred):
    # Slice the final predicted time step from the actual value
    # This represents how many days into the future we're forecasting
    y_actual_last = y_actual[:, -1:]
    y_pred_last = y_pred[:, -1:]

    # Calculate return ratio
    pred_return_ratio = tf.math.divide(tf.math.subtract(y_pred_last, y_actual_last), y_actual_last)
    actual_return_ratio = tf.math.divide(tf.math.subtract(y_actual[:, -2:-1], y_actual[:, -1:]), y_actual[:, -2:-1])
    mse = tf.keras.losses.mean_squared_error(pred_return_ratio, actual_return_ratio)

    # Create an array of all_ones so that we can calculate all permutations of subtractions
    all_ones = tf.ones([y_actual_last.shape[0], 1], dtype=tf.float32)

    # Creates a N x N matrix with every predicted return ratio for each company subtracted with every other
    # company
    pred_dif = tf.math.subtract(
        tf.matmul(pred_return_ratio, all_ones, transpose_b=True),
        tf.matmul(all_ones, pred_return_ratio, transpose_b=True)
    )

    # Creates an N x N matrix containing every actual return ratio for each company subtracted with every other
    # company By switching the order of the all_ones matricies and the actual prices, a negative sign is introduced
    # When RELU is applied later, correct predictions will not affect loss while incorrect predictions will affect
    # loss depending on how incorrect the prediction was
    actual_dif = tf.math.subtract(
        tf.matmul(all_ones, actual_return_ratio, transpose_b=True),
        tf.matmul(actual_return_ratio, all_ones, transpose_b=True)
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

    loss = (tf.cast(1000, tf.float32) * rank_loss) + (mse)
    # loss = (tf.cast(1, tf.float32) * rank_loss)

    return loss

def rank_loss_rr_labels(y_actual, y_pred):
    # Slice the final predicted time step from the actual value
    # This represents how many days into the future we're forecasting
    y_actual_last = y_actual[:, -1:]
    y_pred_last = y_pred[:, -1:]

    # Calculate return ratio
    pred_return_ratio = y_pred_last
    actual_return_ratio = y_actual_last
    mse = tf.keras.losses.mean_squared_error(pred_return_ratio, actual_return_ratio)

    # Create an array of all_ones so that we can calculate all permutations of subtractions
    all_ones = tf.ones([y_actual_last.shape[0], 1], dtype=tf.float32)

    # Creates a N x N matrix with every predicted return ratio for each company subtracted with every other
    # company
    pred_dif = tf.math.subtract(
        tf.matmul(pred_return_ratio, all_ones, transpose_b=True),
        tf.matmul(all_ones, pred_return_ratio, transpose_b=True)
    )

    # Creates an N x N matrix containing every actual return ratio for each company subtracted with every other
    # company By switching the order of the all_ones matricies and the actual prices, a negative sign is introduced
    # When RELU is applied later, correct predictions will not affect loss while incorrect predictions will affect
    # loss depending on how incorrect the prediction was
    actual_dif = tf.math.subtract(
        tf.matmul(all_ones, actual_return_ratio, transpose_b=True),
        tf.matmul(actual_return_ratio, all_ones, transpose_b=True)
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

    loss = (tf.cast(100, tf.float32) * rank_loss) + (mse)
    # loss = (tf.cast(1, tf.float32) * rank_loss)

    return loss

def rank_loss_rr_explicit(y_actual, y_pred):

    # Slice the final predicted time step from the actual value
    # This represents how many days into the future we're forecasting
    y_actual_last = y_actual[:, -1:]
    # Doesn't need to be truncated
    y_pred_last = y_pred



    # Calculate return ratio
    pred_return_ratio = tf.math.divide(tf.math.subtract(y_pred_last, y_actual_last), y_actual_last)
    actual_return_ratio = tf.math.divide(tf.math.subtract(y_actual[:, -2:-1], y_actual[:, -1:]), y_actual[:, -2:-1])
    mse = tf.keras.losses.mean_squared_error(pred_return_ratio, actual_return_ratio)

    # Create an array of all_ones so that we can calculate all permutations of subtractions
    all_ones = tf.ones([y_actual_last.shape[0], 1], dtype=tf.float32)

    # Creates a N x N matrix with every predicted return ratio for each company subtracted with every other
    # company
    pred_dif = tf.math.subtract(
        tf.matmul(pred_return_ratio, all_ones, transpose_b=True),
        tf.matmul(all_ones, pred_return_ratio, transpose_b=True)
    )

    # Creates an N x N matrix containing every actual return ratio for each company subtracted with every other
    # company By switching the order of the all_ones matricies and the actual prices, a negative sign is introduced
    # When RELU is applied later, correct predictions will not affect loss while incorrect predictions will affect
    # loss depending on how incorrect the prediction was
    actual_dif = tf.math.subtract(
        tf.matmul(all_ones, actual_return_ratio, transpose_b=True),
        tf.matmul(actual_return_ratio, all_ones, transpose_b=True)
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

    loss = (tf.cast(1000, tf.float32) * rank_loss) + (mse)
    # loss = (tf.cast(1, tf.float32) * rank_loss)

    return loss

# These are the third generation
def rank_loss_correct_pred_rr(y_actual, y_pred):

    # Calculate return ratio
    pred_return_ratio = tf.math.divide(tf.math.subtract(y_pred[:, -1:None], y_actual[:, -2:-1]), y_actual[:, -2:-1])
    actual_return_ratio = tf.math.divide(tf.math.subtract(y_actual[:, -1:None], y_actual[:, -2:-1]), y_actual[:, -2:-1])
    mse = tf.keras.losses.mean_squared_error(pred_return_ratio, actual_return_ratio)

    # Create an array of all_ones so that we can calculate all permutations of subtractions
    all_ones = tf.ones([y_actual[:, -1:None].shape[0], 1], dtype=tf.float32)

    # Creates a N x N matrix with every predicted return ratio for each company subtracted with every other
    # company
    pred_dif = tf.math.subtract(
        tf.matmul(pred_return_ratio, all_ones, transpose_b=True),
        tf.matmul(all_ones, pred_return_ratio, transpose_b=True)
    )

    # Creates an N x N matrix containing every actual return ratio for each company subtracted with every other
    # company By switching the order of the all_ones matricies and the actual prices, a negative sign is introduced
    # When RELU is applied later, correct predictions will not affect loss while incorrect predictions will affect
    # loss depending on how incorrect the prediction was
    actual_dif = tf.math.subtract(
        tf.matmul(all_ones, actual_return_ratio, transpose_b=True),
        tf.matmul(actual_return_ratio, all_ones, transpose_b=True)
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

    loss = (tf.cast(1000, tf.float32) * rank_loss) + (mse)
    # loss = (tf.cast(1, tf.float32) * rank_loss)

    return loss

# These are the ones we've mainly used
def rank_loss_correct_pred_return_sequences(y_actual, y_pred):

    y_actual = tf.cast(y_actual, tf.float16)
    y_pred = tf.cast(y_pred, tf.float16)

    # Calculate return ratio
    actual_return_ratio = tf.math.divide(tf.math.subtract(y_actual[:, 1:], y_actual[:, 0:-1]), y_actual[:, 0:-1])
    pred_return_ratio = tf.math.divide(tf.math.subtract(y_pred[:, 1:], y_actual[:, 0:-1]), y_actual[:, 0:-1])
    mse = tf.keras.losses.mean_squared_error(pred_return_ratio, actual_return_ratio)
    mse = tf.reduce_mean(mse)

    # tf.print(pred_return_ratio.dtype, output_stream=sys.stdout)

    p = (tf.repeat(tf.expand_dims(pred_return_ratio, 2), pred_return_ratio.shape[0], axis=2))
    pred_dif = (p - tf.transpose(p))

    p = (tf.repeat(tf.expand_dims(actual_return_ratio, 2), actual_return_ratio.shape[0], axis=2))
    actual_dif = -(p - tf.transpose(p))

    # Calculate all the different (ri-rj) * (ri-rj)hat sets through time)
    s = tf.multiply(pred_dif, actual_dif)
    # Remove all negative values
    s = tf.nn.relu(s)
    # Average across time
    s = tf.reduce_mean(s, 1)
    # Average all values
    s = tf.reduce_mean(s)

    loss = (tf.cast(1000, tf.float16) * s) + mse
    # loss = (tf.cast(1, tf.float32) * rank_loss)

    return loss


# Define a custom layer in Tensorflow 2.0 to implement a matrix multiplication
# operation using the output of an LSTM layer and another given matrix
class Ein_Multiply(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(Ein_Multiply, self).__init__()

    def call(self, input_m, tag):
        # What we've been using that works
        # return tf.einsum('ij,ik->ik', input[0], input[1])
        # What I think the result SHOULD be using
        # return tf.einsum('ij,jk->ik', input[0], input[1])

        # This is just multipling the LAST time step by the information
        # Cannot use because of item assignment
        # input[1][:, -1, :] = tf.einsum('ij,ik->ik', input[0], input[1][:, -1, :])
        # return input[1]

        # Yunings first suggestion, where the time-dimension is left the same size, but the multiplation occurs on the
        # outer layer
        return tf.einsum(tag, input_m[0], input_m[1])

# time dimension tag: 'mn,ntd->mtd'

# Create a function for the Tensorflow implementation of Tensorflow 2
def leaky_relu(x):
    return tf.keras.layers.LeakyReLU(alpha=0.2)(x)


class TF_Models(Graph_Entities):
    def __init__(self, data_path, models_path, reload=False, time_major=False):

        self.time_major = time_major

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.data_path = data_path
        self.models_path = models_path

        if not reload:
            self.load_obj = self.load_data_and_labels()
        else:
            self.load_obj = None

        if self.load_obj is None:
            # Re-use the methods from explore_entities
            self.entities, self.entities_idx = super()._generate_list_of_entities(data_path)
            self.relations_dict = super()._generate_relations()
            self.Normalized_Adjacency_Matrix = super()._generate_normalized_ajacency_matrix()

            # # Expanding Dimensions
            # self.Normalized_Adjacency_Matrix = tf.expand_dims(self.Normalized_Adjacency_Matrix, axis=1)
            # input("")

            self.XX_tf = self._load_data_into_TF()
            self.YY_tf = self._create_labels()
            # Create the ground truth return ratios
            self.RR_tf = tf.divide(tf.subtract(self.YY_tf, self.XX_tf[:, :, 0]), self.XX_tf[:, :, 0])

            self.save_data_and_labels()
        else:
            # If we've calculated this before, just load in a pickle file
            self.entities = self.load_obj['entities']
            self.entities_idx = self.load_obj['entities_idx']
            self.relations_dict = self.load_obj['relations_dict']
            self.Normalized_Adjacency_Matrix = self.load_obj['Normalized_Adjacency_Matrix']
            self.XX_tf = self.load_obj['XX_tf']
            self.YY_tf = self.load_obj['YY_tf']
            self.RR_tf = self.load_obj['RR_tf']

        # Stores the last model affected in memory
        self.model = None
        self.epochs_n = 0
        self.tag_t = ''
        self.loss_t = None
        self.date_t = None
        self.hidden_units = None
        self.model_name = None

        self.history = None
        self.schedule_function = None
        self.increasing = True

        # Automatically splits the data
        self.data_splits = self.split_data()

    '''Returns a 3 Dimensional Tensor with dimensions: (Entities, Sequences, Features)'''
    '''For the LSTM model, the first dimension is a placeholder, so entities must be first'''

    def _load_data_into_TF(self):

        # Start the loading bar by initializing it
        bar = widgets.IntProgress(min=0, max=len(self.entities), value=0,
                                  layout=Layout(width='auto'))
        text = widgets.Text(value='Loading Entities into Memory:', description='', disabled=True,
                            layout=Layout(width='auto'))
        loading_bar = GridBox(children=[text, bar], layout=Layout(width='auto'))
        display(loading_bar)

        XX_t = []
        for ent in self.entities:
            df = pd.read_csv(self.data_path + '/' + ent + '.csv')

            # Swap the 30 day and 1 day differences (The original dataset has these columns swapped
            closes = df[['30D MA']].values
            thirty_ma = df[['Close']].values

            df[['Close']] = closes
            df[['30D MA']] = thirty_ma

            # Append the 2x2 tensor of each company's features through all time-sequences to the main tensor
            XX_t.append(df[0:].values)

            # Increment the loading bar
            bar.value += 1

        XX_t = tf.constant(XX_t, dtype=tf.float32)

        loading_bar.close()
        return XX_t

    '''This function should be modified depending on the specific use case. In IFPTSND, the labels should be
        the return ratio of day t+1. However, that is project specific and should be changed if financial market
        predictions are the final labels'''

    def _create_labels(self):

        # Start the loading bar by initializing it
        bar = widgets.IntProgress(min=0, max=len(self.entities), value=0,
                                  layout=Layout(width='auto'))
        text = widgets.Text(value='Calculating Labels:', description='', disabled=True, layout=Layout(width='auto'))
        loading_bar = GridBox(children=[text, bar], layout=Layout(width='auto'))
        display(loading_bar)

        XX_t = self.XX_tf.numpy()

        # Labels are currently the same day close price
        YY_t = XX_t[:, :, 0]
        # Labels are now the next day close price
        YY_t = YY_t[:, 1:]
        # Now that the number of time steps don't match between XX and YY,
        # the last day of XX_tf needs to be removed since we don't have a prediction for it
        self.XX_tf = tf.constant(XX_t[:, 0:-1, :])

        # According to paper #2, the normalized price of each stock should be the output, not the return ratio
        '''
        # In our project, the labels shouldn't be the actual price of the next day. It needs to be the return ratio
        # had we purchased the stock
        YY_tf = np.copy(YY_t)
        for e in range(XX_t.shape[0]):
            # Increment the loading bar
            bar.value += 1
            for t in range(XX_t.shape[1] - 1):
                YY_tf[e, t] = (YY_t[e, t] - XX_t[e, t, 0]) / XX_t[e, t, 0]

                # Some days the price won't change between days
                # Those zeroes need to be made really small, but not actually 0 to avoid loss errors later
                if YY_tf[e, t] == 0:
                    YY_tf[e, t] = 1e-10
        '''

        loading_bar.close()
        return tf.constant(YY_t)

    '''Splits the data into a training, validation, and testing set. The proportion of entries in each category
        can be set using the list. For example by default, 60% training, 15% validation, 25% testing. Axis specifies
        which dimension to make the split. In the default example, it's time'''
    '''returns a dictionary containing the splits'''

    def split_data(self, axis=1, perc_split=[55, 25, 20]):

        # Given a total and list of splits, evenly distributes the total amount proportional to the given list
        def split_windows(total, percentages_list):
            # Get a sum of the initial total
            percentage_sum = sum(percentages_list)

            # Create a new list based on a percentage of the total
            new_splits = []
            for perc in percentages_list:
                new_splits.append(int(total * (perc / percentage_sum)))

            if sum(new_splits) != total:
                new_splits[-1] = new_splits[-1] + (total - sum(new_splits))

            return new_splits

        x_train, x_val, x_test = tf.split(self.XX_tf, split_windows(self.XX_tf.shape[1], perc_split),
                                          axis=axis)
        y_train, y_val, y_test = tf.split(self.YY_tf, split_windows(self.YY_tf.shape[1], perc_split),
                                          axis=axis)
        rr_train, rr_val, rr_test = tf.split(self.RR_tf, split_windows(self.YY_tf.shape[1], perc_split),
                                             axis=axis)

        return {'x_train': x_train, 'y_train': y_train, 'rr_train': rr_train,
                'x_val': x_val, 'y_val': y_val, 'rr_val': rr_val,
                'x_test': x_test, 'y_test': y_test, 'rr_test': rr_test}

    '''Displays a dropdown selection of model parameters to select from
        When the generate button is pressed, that model is saved as a part
        of the class to be used in training'''

    def generate_model(self):
        model_drop = widgets.Dropdown(
            value='lstm',
            options=['lstm', 'lstm_gcn_1', 'lstm_gcn_2', 'lstm_gcn_3', 'seperate_lstm_gcn_3', 'seperate_lstm_gcn_1'],
            description='Model Types:',
            style=dict(description_width='initial'),
        )

        loss_drop = widgets.Dropdown(
            value='mse',
            options=['mse', 'rank_loss', 'custom_mse'],
            description='Loss Functions:',
            style=dict(description_width='initial'),
        )

        act_drop = widgets.Dropdown(
            options=['relu', 'leaky_relu', 'sigmoid'],
            description='Activation Functions:',
            style=dict(description_width='initial'),
        )

        units = widgets.BoundedIntText(
            value=64,
            min=16,
            max=256,
            description='Number of Units',
            style=dict(description_width='initial'),
        )

        butt_random = widgets.Checkbox(
            value=False,
            description="Random Seed",
            style=dict(description_width='initial'),
        )

        button = widgets.Button(description="Generate")

        container = GridBox(children=[model_drop, loss_drop, act_drop, units, button, butt_random])
        display(container)

        gb = self._generate_model

        def on_button_clicked(b):
            gb(model_drop.value, loss_drop.value,
               act_drop.value, units.value, butt_random)

        button.on_click(on_button_clicked)

    def _generate_model(self, model_type, loss_function, activation, hidden_units, true_random,
                        learning_rate=1e-5, decay_rate=1e-6, alpha=1):

        if os.path.exists('./tmp'):
            shutil.rmtree('./tmp')

        self.model = None
        if './tmp' in os.listdir('./'):
            os.rmdir('./tmp')

        # Useful later
        self.model_type = model_type
        self.epochs_n = 0

        if activation == 'leaky_relu':
            activation = leaky_relu

        if model_type != 'lstm':
            gcn_shape = True
        else:
            gcn_shape = False

        # By controlling the random starting weights, the models can be more accurately assessed against one another
        if not true_random:
            tf.random.set_seed(1337)

        # Must be the same between runs
        tf.random.set_seed(1337)
        tf.random.set_seed(420)
        tf.random.set_seed(17)
        tf.random.set_seed(100998)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam'
        )

        '''Types of model inputs'''
        # Sequential Input Data

        # input_seq = keras.Input(shape=self.XX_tf.shape[1:])
        # Originally XX_tf would always be the maximum size, but now we need it to work with split data that doesn't contain the whole
        input_seq = keras.Input(shape=(
            self.data_splits['x_train'].shape[1] + self.data_splits['x_val'].shape[1] +
            self.data_splits['x_test'].shape[1],
            self.data_splits['x_train'].shape[2]))

        # input_seq = keras.Input(shape=self.XX_tf.shape[1:])
        # Originally XX_tf would always be the maximum size, but now we need it to work with split data that doesn't contain the whole
        input_seq = keras.Input(shape=(
            self.data_splits['x_train'].shape[1],
            self.data_splits['x_train'].shape[2]))
      

        # Normalized Ajacency Matrix
        if gcn_shape:
            input_rel = keras.Input(shape=self.Normalized_Adjacency_Matrix.shape[0])

        # One LSTM layer with return sequences, One LSTM without return sequences, One Dense Layer: (None, 1)
        if model_type == 'lstm':
            do = 0.333
            x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=do, recurrent_dropout=do)(input_seq)
            x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=do, recurrent_dropout=do)(x)
            x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=do, recurrent_dropout=do)(x)
            x = LSTM(hidden_units, return_sequences=False, activation=activation, dropout=do, recurrent_dropout=do)(x)
            # x = LSTM(hidden_units, return_sequences=False, activation=activation, dropout=0.2)(x)
            # x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=0.5,
            #          recurrent_dropout=0.5)(x)
            # x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=0.5,
            #          recurrent_dropout=0.5)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq], outputs=x)

        # Two LSTM layers, One GCN Layer, One Dense Layer: (None, 1)
        elif model_type == 'lstm_gcn_1':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

        # Two LSTM layers, Two GCN Layers, One Dense Layer: (None, 1)
        elif model_type == 'lstm_gcn_2':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

        # Two LSTM layers, Three GCN Layers, One Dense Layer: (None, 1)
        elif model_type == 'lstm_gcn_3':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

        elif model_type == 'seperate_lstm_gcn_3':
            # Load in an already trained LSTM Model
            file_name = '01-04-2021--13--38--LSTM-3e-5LR--75Epochs--mse-Loss--64-HU--'
            model_path = './models'
            pre_trained_lstm = tf.keras.models.load_model(model_path + f'/{file_name}', compile=False,
                                                          custom_objects={'leaky_relu': leaky_relu})

            # Change the names to avoid conflicts
            pre_trained_lstm.layers[1]._name = 'b'
            pre_trained_lstm.layers[2]._name = 'c'

            # All of the layers from the loaded in LSTM model layers are used except for the last output and first input
            # This takes the place of the LSTM layers since we want the LSTM and GCN to be separately trained
            x = pre_trained_lstm.layers[1](input_seq)
            x = pre_trained_lstm.layers[2](x)

            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

            # Make sure that the weights for the lstm model cannot be updated
            self.model.layers[1].trainable = False
            self.model.layers[3].trainable = False

        elif model_type == 'seperate_lstm_gcn_1':
            # Load in an already trained LSTM Model
            file_name = '01-04-2021--13--38--LSTM-3e-5LR--75Epochs--mse-Loss--64-HU--'
            model_path = './models'
            pre_trained_lstm = tf.keras.models.load_model(model_path + f'/{file_name}', compile=False,
                                                          custom_objects={'leaky_relu': leaky_relu})

            # Change the names to avoid conflicts
            pre_trained_lstm.layers[1]._name = 'b'
            pre_trained_lstm.layers[2]._name = 'c'

            # All of the layers from the loaded in LSTM model layers are used except for the last output and first input
            # This takes the place of the LSTM layers since we want the LSTM and GCN to be separately trained
            x = pre_trained_lstm.layers[1](input_seq)
            x = pre_trained_lstm.layers[2](x)

            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

            # Make sure that the weights for the lstm model cannot be updated
            self.model.layers[1].trainable = False
            self.model.layers[3].trainable = False
        else:
            sys.exit('You must specify a model type')

        # Specify which loss function to use
        if loss_function == 'rank_loss':
            loss_function = rank_loss_func(rr_train=self.data_splits['rr_train'], rr_val=self.data_splits['rr_val'],
                                           alpha=alpha, beta=1)
        elif loss_function == 'custom_mse':
            loss_function = None

        self.model.compile(loss=loss_function, optimizer=optimizer)
        self.model.summary()

        # Save the tags in memory for saving the file
        self.hidden_units = hidden_units
        self.model_type = model_type

        # If the loss function is not a string, get its name directly
        if type(loss_function) is not str:
            loss_function = loss_function.__name__
        self.loss_t = loss_function

    def _generate_model_split_batch(self, model_type, loss_function, activation, hidden_units, batch_splits, true_random=False,
                        learning_rate=1e-5, decay_rate=1e-6, alpha=1):

        if os.path.exists('./tmp'):
            shutil.rmtree('./tmp')

        self.model = None
        if './tmp' in os.listdir('./'):
            os.rmdir('./tmp')

        # Useful later
        self.model_type = model_type
        self.epochs_n = 0

        if activation == 'leaky_relu':
            activation = leaky_relu

        if model_type != 'lstm':
            gcn_shape = True
        else:
            gcn_shape = False

        # By controlling the random starting weights, the models can be more accurately assessed against one another
        if not true_random:
            tf.random.set_seed(1337)

        # Must be the same between runs
        tf.random.set_seed(1337)
        tf.random.set_seed(420)
        tf.random.set_seed(17)
        tf.random.set_seed(100998)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam'
        )

        '''Types of model inputs'''
        # Sequential Input Data

        # input_seq = keras.Input(shape=self.XX_tf.shape[1:])
        # Originally XX_tf would always be the maximum size, but now we need it to work with split data that doesn't contain the whole
        input_seq = keras.Input(shape=(batch_splits['x_train_t'].shape[1], batch_splits['x_train_t'].shape[2]))

        # Normalized Ajacency Matrix
        if gcn_shape:
            input_rel = keras.Input(shape=self.Normalized_Adjacency_Matrix.shape[0])

        # One LSTM layer with return sequences, One LSTM without return sequences, One Dense Layer: (None, 1)
        if model_type == 'lstm':
            x = LSTM(hidden_units, return_sequences=False, activation=activation, dropout=0.1,
                     recurrent_dropout=0.1)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation, dropout=0.1,
                     recurrent_dropout=0.1)(x)
            # x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=0.5,
            #          recurrent_dropout=0.5)(x)
            # x = LSTM(hidden_units, return_sequences=True, activation=activation, dropout=0.5,
            #          recurrent_dropout=0.5)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq], outputs=x)

        # Two LSTM layers, One GCN Layer, One Dense Layer: (None, 1)
        elif model_type == 'lstm_gcn_1':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

        # Two LSTM layers, Two GCN Layers, One Dense Layer: (None, 1)
        elif model_type == 'lstm_gcn_2':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

        # Two LSTM layers, Three GCN Layers, One Dense Layer: (None, 1)
        elif model_type == 'lstm_gcn_3':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

        elif model_type == 'seperate_lstm_gcn_3':
            # Load in an already trained LSTM Model
            file_name = '01-04-2021--13--38--LSTM-3e-5LR--75Epochs--mse-Loss--64-HU--'
            model_path = './models'
            pre_trained_lstm = tf.keras.models.load_model(model_path + f'/{file_name}', compile=False,
                                                          custom_objects={'leaky_relu': leaky_relu})

            # Change the names to avoid conflicts
            pre_trained_lstm.layers[1]._name = 'b'
            pre_trained_lstm.layers[2]._name = 'c'

            # All of the layers from the loaded in LSTM model layers are used except for the last output and first input
            # This takes the place of the LSTM layers since we want the LSTM and GCN to be separately trained
            x = pre_trained_lstm.layers[1](input_seq)
            x = pre_trained_lstm.layers[2](x)

            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

            # Make sure that the weights for the lstm model cannot be updated
            self.model.layers[1].trainable = False
            self.model.layers[3].trainable = False

        elif model_type == 'seperate_lstm_gcn_1':
            # Load in an already trained LSTM Model
            file_name = '01-04-2021--13--38--LSTM-3e-5LR--75Epochs--mse-Loss--64-HU--'
            model_path = './models'
            pre_trained_lstm = tf.keras.models.load_model(model_path + f'/{file_name}', compile=False,
                                                          custom_objects={'leaky_relu': leaky_relu})

            # Change the names to avoid conflicts
            pre_trained_lstm.layers[1]._name = 'b'
            pre_trained_lstm.layers[2]._name = 'c'

            # All of the layers from the loaded in LSTM model layers are used except for the last output and first input
            # This takes the place of the LSTM layers since we want the LSTM and GCN to be separately trained
            x = pre_trained_lstm.layers[1](input_seq)
            x = pre_trained_lstm.layers[2](x)

            x = Ein_Multiply()([input_rel, x])
            x = Dense(hidden_units, activation=activation)(x)
            x = Dense(1, activation=activation)(x)
            self.model = tf.keras.Model(inputs=[input_seq, input_rel], outputs=x)

            # Make sure that the weights for the lstm model cannot be updated
            self.model.layers[1].trainable = False
            self.model.layers[3].trainable = False
        else:
            sys.exit('You must specify a model type')

        # Specify which loss function to use
        if loss_function == 'rank_loss':
            loss_function = rank_loss_func(rr_train=self.data_splits['rr_train'], rr_val=self.data_splits['rr_val'],
                                           alpha=alpha)
        elif loss_function == 'custom_mse':
            loss_function = None

        self.model.compile(loss=loss_function, optimizer=optimizer)
        self.model.summary()

        # Save the tags in memory for saving the file
        self.hidden_units = hidden_units
        self.model_type = model_type

        # If the loss function is not a string, get its name directly
        if type(loss_function) is not str:
            loss_function = loss_function.__name__
        self.loss_t = loss_function

    '''Run this function to train the generated model. Using early stopping, Model Checkpoint as callbacks, and the
        learning rate scheduler, this training can be run multple times on the same model with different learning rates.
        Additionally, the best model will be selected based on the validation set to avoid worry of over-training'''

    def train_model_loop(self, epoch_batches, learning_rate=5e-5, one_loop_only=False):
        # Adjust the sizes of the matrix so that Google Colab can accept validation and test sets of different time-steps
        amt_to_split = self.data_splits['x_train'].shape[1] - self.data_splits['x_val'].shape[1]
        x_val = tf.keras.layers.ZeroPadding1D(padding=(amt_to_split,0))(self.data_splits['x_val'])
        z = tf.zeros([self.data_splits['y_val'].shape[0], amt_to_split], tf.float32)
        y_val = tf.concat([z, self.data_splits['y_val']], 1)  

        def model_continue_check(history, increasing):
            print("#" * 125)
            last_epochs = history.history['val_loss']
            last_lr = history.history['lr'][0]

            # For the first epoch set we can set the value very high for training and then be leveled off
            if last_lr > 1e-3:
                last_lr = 5e-4

            # If we reached NaN for any reason
            if np.isnan(np.sum(last_epochs)):
                return None, None, False

            e_max = np.max(last_epochs)
            e_min = np.min(last_epochs)

            if last_epochs[0] == e_min and not increasing:
              return None, None, False

            # In the case were learning_rate might be too slow,
            # Continue the loop, increase the learning rate
            if last_epochs[-1] == e_min:
                new_lr = last_lr * 1.3
                print(f'Increasing learning rate to: {"{:.3e}".format(new_lr)}')

                def scheduler(epoch, lr):
                    return new_lr

                return True, scheduler, True

            # If the last item was not the minimum, but the min and max value are VERY close to each other
            # Then the bottom has probably been found
            if (abs(e_max - e_min) / e_max) <= 0.01:
                  return None, None, False

            # In the case were learning_rate might be too fast,
            # Continue the loop, decrease the learning rate
            if last_epochs[-1] != np.min(last_epochs):
                new_lr = last_lr * 0.3333

                print(f'Decreasing learning rate to: {"{:.3e}".format(new_lr)}')

                def scheduler(epoch, lr):
                    return new_lr

                return False, scheduler, True

            # In some weird case that is unaccounted for, stop training
            print('######Error?: No Statement Reached, Training Halted######')
            return None, None, False

        # New Tensorboard Code
        log_dir = "logs/fit/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + f'--{self.loss_t}-Loss--{self.hidden_units}-HU--{self.tag_t}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

        # If the validation loss doesn't improve after X epochs, stop training
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, mode='min')

        # # Together these functions allow the train_model feature to update the learning_rate without
        # # re-establishing the model
        def scheduler(epoch, lr):
            return learning_rate

        self.schedule_function = scheduler

        # RLROP = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss', factor=0.8, patience=1, verbose=0,
        #     mode='min', min_delta=0.0001, cooldown=1000, min_lr=1e-6,
        # )

        # Callback for loading the best validation loss checkpoint
        checkpoint_filepath = './tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        inputs_train = self.data_splits['x_train']
        inputs_val = x_val
        
        if self.model_type != 'lstm':
            inputs_train.append(self.Normalized_Adjacency_Matrix)
            inputs_val.append(self.Normalized_Adjacency_Matrix)

        loop = True
        while loop:
            # If the dimensions need to be evaluated
            # print(inputs_train.shape)
            # print(train_labels.shape)
            # print(inputs_val.shape)
            # print(val_labels.shape)
            # input('Wait')
            try:
                self.history = self.model.fit(inputs_train,
                                              self.data_splits['y_train'],
                                              batch_size=int(self.data_splits['x_train'].shape[0]),
                                              epochs=epoch_batches,
                                              validation_data=(inputs_val, y_val),
                                              callbacks=[model_checkpoint_callback,
                                                         tensorboard_callback,
                                                         tf.keras.callbacks.LearningRateScheduler(self.schedule_function,
                                                                                                  verbose=0)])
            except():
                print("Ooopsies")

            # Keep track of the number of epochs we've trained
            self.epochs_n = len(self.history.history['val_loss']) + self.epochs_n

            # Reloads the checkpoint with the lowest validation loss
            self.model.load_weights(checkpoint_filepath)

            # Determines logic behind increasing or decreasing learning rate
            self.increasing, self.schedule_function, loop = model_continue_check(self.history, self.increasing)

            # Just in case something has gone wrong the schedule function algorithm, there needs to be an escape case
            if self.epochs_n > 350 or one_loop_only:
                loop = False

        self.date_t = datetime.datetime.now().strftime("%m-%d-%Y--%H--%M")

        return self.model

    '''This version can be passed explicit types of data for structuring purposes.'''
    def train_model_loop_batch_splits(self, epoch_batches, batch_splits=None, learning_rate=5e-5, one_loop_only=False):

        def model_continue_check(history, last_hoorah):
            print("#" * 125)
            last_epochs = history.history['val_loss']
            last_lr = history.history['lr'][0]

            e_max = np.max(last_epochs)
            e_min = np.min(last_epochs)

            # In the case were learning_rate might be too slow,
            # Continue the loop, increase the learning rate
            if last_epochs[-1] == e_min:
                new_lr = last_lr * 1.3
                print(f'Increasing learning rate to: {"{:.3e}".format(new_lr)}')

                def scheduler(epoch, lr):
                    return new_lr

                return last_hoorah, scheduler, True

            # If the last item was not the minimum, but the min and max value are VERY close to each other
            # Then the bottom has probably been found
            if (abs(e_max - e_min) / e_max) <= 0.01 or last_epochs[0] == e_min:

                # Changed my mind on Last Hoorah, removing
                last_hoorah = False

                if last_hoorah:
                    print('Attempting Large LR Increase in case bad Minimum Found')
                    new_lr = last_lr * 10

                    def scheduler(epoch, lr):
                        return new_lr

                    return False, scheduler, True
                # Otherwise, end the loop
                else:
                    return None, None, False

            # In the case were learning_rate might be too fast,
            # Continue the loop, decrease the learning rate
            if last_epochs[-1] != np.min(last_epochs):
                new_lr = last_lr * 0.3333

                # Unless we're on the last hoorah, in which case the large increase from earlier needs to be removed
                new_lr = last_lr / 10

                print(f'Decreasing learning rate to: {"{:.3e}".format(new_lr)}')

                def scheduler(epoch, lr):
                    return new_lr

                # If the best epoch is only the first in the list, then the hoorah failed
                if last_epochs[0] == np.min(last_epochs):
                    newrah = False
                else:
                    newrah = True
                return newrah, scheduler, True

            # In some weird case that is unaccounted for, stop training
            print('######Error?: No Statement Reached, Training Halted######')
            return None, None, False

        # New Tensorboard Code
        log_dir = "logs/fit/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + f'--{self.loss_t}-Loss--{self.hidden_units}-HU--{self.tag_t}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # If the validation loss doesn't improve after X epochs, stop training
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, mode='min')

        # # Together these functions allow the train_model feature to update the learning_rate without
        # # re-establishing the model
        def scheduler(epoch, lr):
            return learning_rate

        self.schedule_function = scheduler

        # RLROP = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss', factor=0.8, patience=1, verbose=0,
        #     mode='min', min_delta=0.0001, cooldown=1000, min_lr=1e-6,
        # )

        # Callback for loading the best validation loss checkpoint
        checkpoint_filepath = './tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        inputs_train = batch_splits['x_train_t']
        inputs_val = batch_splits['x_train_v']
        if self.model_type != 'lstm':
            inputs_train.append(self.Normalized_Adjacency_Matrix)
            inputs_val.append(self.Normalized_Adjacency_Matrix)

        loop = True
        while loop:
            # If the dimensions need to be evaluated
            # print(inputs_train.shape)
            # print(train_labels.shape)
            # print(inputs_val.shape)
            # print(val_labels.shape)
            # input('Wait')
            self.history = self.model.fit(inputs_train,
                                          batch_splits['y_train_y'],
                                          batch_size=int(batch_splits['x_train_t'].shape[0]),
                                          epochs=epoch_batches,
                                          validation_data=(inputs_val, batch_splits['y_train_v']),
                                          callbacks=[model_checkpoint_callback,
                                                     tensorboard_callback,
                                                     tf.keras.callbacks.LearningRateScheduler(self.schedule_function,
                                                                                              verbose=0)])

            # Keep track of the number of epochs we've trained
            self.epochs_n = len(self.history.history['loss']) + self.epochs_n

            # Reloads the checkpoint with the lowest validation loss
            self.model.load_weights(checkpoint_filepath)

            # Determines logic behind increasing or decreasing learning rate
            self.last_hoorah, self.schedule_function, loop = model_continue_check(self.history, self.last_hoorah)

            # Just in case something has gone wrong the schedule function algorithm, there needs to be an escape case
            if self.epochs_n > 350 or one_loop_only:
                loop = False

        self.date_t = datetime.datetime.now().strftime("%m-%d-%Y--%H--%M")

        return self.model


    '''Saves the model that is currently loaded to the specified directory. Tags can be given to the model and the
        filename utilizes the parameters from training and time of training to create the model name'''

    def save_model(self, tag=''):
        self.model_name = f'{self.date_t}-{tag}-{self.epochs_n}Epochs-{self.loss_t}-Loss-{self.hidden_units}-HU-{self.tag_t}'
        self.model.save(self.models_path + f'/{self.model_name}')
        return self.model_name

    '''After parsing the .csv and .json data and then converting into data accessible by Tensorflow, it can be pickled
        and reloaded next time a model needs to be trained. On Google Colab specifically, this can save ~10 minutes 
        from a session restart'''

    def save_data_and_labels(self):
        obj = {'XX_tf': self.XX_tf, 'YY_tf': self.YY_tf, 'entities': self.entities, 'entities_idx': self.entities_idx,
               'relations_dict': self.relations_dict, 'RR_tf': self.RR_tf,
               'Normalized_Adjacency_Matrix': self.Normalized_Adjacency_Matrix}

        pickle.dump(obj, open(self.data_path + fr'/parsed_data.p', 'wb'))

    '''Given that a save_reload file was created already in the directory, the model will attempt to load it'''

    def load_data_and_labels(self):
        files = [f for f in os.listdir(self.data_path) if isfile(join(self.data_path, f))]
        files = [f for f in files if f.endswith('.p')]
        if len(files) > 1:
            sys.exit('There are multiple saved obj files in the data_path. Delete all but one or start over')
        if len(files) < 1:
            return None
        with open(self.data_path + fr'/parsed_data.p', 'rb') as read:
            return pickle.load(read)