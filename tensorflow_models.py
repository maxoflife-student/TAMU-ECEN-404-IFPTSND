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

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def custom_mse(y_actual, y_pred):

    squared_error = math.ops.squared_difference(y_actual, y_pred)
    average_mse = kb.mean(squared_error, axis=-1)

    return average_mse

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

        # Divide the rank_loss by the total time-steps to equalize loss if the given time-steps are different
        rank_loss = tf.divide(rank_loss, tf.constant(y_actual.shape[1], dtype=tf.float32))

        # Take the squared difference between the y_actual and y_pred
        squared_difference = math_ops.squared_difference(y_actual, y_pred)
        # Average each of values, taking the magnitude
        sd_mean = kb.mean(squared_difference, axis=-1)

        loss = tf.cast(alpha, tf.float32) * rank_loss + sd_mean
        return loss

    # Return the RLF function instance
    return rlf


# Define a custom layer in Tensorflow 2.0 to implement a matrix multiplication
# operation using the output of an LSTM layer and another given matrix
class Ein_Multiply(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(Ein_Multiply, self).__init__()

    def call(self, input):
        return tf.einsum('ij,ik->ik', input[0], input[1])


# Create a function for the Tensorflow implementation of Tensorflow 2
def leaky_relu(x):
    return tf.keras.layers.LeakyReLU(alpha=0.2)(x)


class TF_Models(Graph_Entities):
    def __init__(self, data_path, models_path, reload=False):

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
            self.XX_tf = self._load_data_into_TF()
            self.YY_tf = self._create_labels()

            self.save_data_and_labels()
        else:
            # If we've calculated this before, just load in a pickle file
            self.entities = self.load_obj['entities']
            self.entities_idx = self.load_obj['entities_idx']
            self.relations_dict = self.load_obj['relations_dict']
            self.Normalized_Adjacency_Matrix = self.load_obj['Normalized_Adjacency_Matrix']
            self.XX_tf = self.load_obj['XX_tf']
            self.YY_tf = self.load_obj['YY_tf']

        # Stores the last model affected in memory
        self.model = None
        self.epochs_n = 0
        self.tag_t = ''
        self.loss_t = None
        self.date_t = None
        self.hidden_units = None

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
            # Append the 2x2 tensor of each company's features through all time-sequences to the main tensor
            XX_t.append(df[0:].values)

            # Increment the loading bar
            bar.value += 1

        XX_t = tf.constant(XX_t)
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

        loading_bar.close()
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
            options=['mse', 'rank_loss'],
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
            value=True,
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
                        learning_rate=1e-5, decay_rate=1e-6):

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

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate)

        '''Types of model inputs'''
        # Sequential Input Data
        input_seq = keras.Input(shape=self.XX_tf.shape[1:])
        # Normalized Ajacency Matrix
        if gcn_shape:
            input_rel = keras.Input(shape=self.Normalized_Adjacency_Matrix.shape[0])

        '''Keras Layers'''
        LSTM_t = LSTM(hidden_units, return_sequences=True, activation=activation)
        LSTM_f = LSTM(hidden_units, return_sequences=False, activation=activation)
        Dense_u = Dense(hidden_units, activation=activation)
        Dense_o = Dense(1, activation=activation)

        '''Custom Layers'''
        # Ein_Multiply

        # One LSTM layer with return sequences, One LSTM without return sequences, One Dense Layer: (None, 1)
        if model_type == 'lstm':
            x = LSTM(hidden_units, return_sequences=True, activation=activation)(input_seq)
            x = LSTM(hidden_units, return_sequences=False, activation=activation)(x)
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
            loss_function = rank_loss_func(1)

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

    def train_model(self, epochs, learning_rate=1e-5):

        self.epochs_n += epochs

        # If the validation loss doesn't improve after 3 epochs, stop training
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

        # Together these functions allow the train_model feature to update the learning_rate without
        # re-establishing the model
        def scheduler(epoch, lr):
            return learning_rate

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            scheduler, verbose=0
        )

        # Callback for loading the best validation loss checkpoint
        checkpoint_filepath = './tmp/checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        inputs_train = [self.data_splits['x_train']]
        inputs_val = [self.data_splits['x_val']]
        if self.model_type != 'lstm':
            inputs_train.append(self.Normalized_Adjacency_Matrix)
            inputs_val.append(self.Normalized_Adjacency_Matrix)

        self.model.fit(inputs_train,
                       self.data_splits['y_train'],
                       batch_size=self.data_splits['x_train'].shape[0],
                       epochs=epochs,
                       validation_data=(inputs_val, self.data_splits['y_val']),
                       callbacks=[early_stopping, model_checkpoint_callback,
                                  lr_schedule])

        # Reloads the checkpoint with the lowest validation loss
        self.model.load_weights(checkpoint_filepath)
        self.date_t = datetime.datetime.now().strftime("%m-%d-%Y--%H--%M")

    '''Saves the model that is currently loaded to the specified directory. Tags can be given to the model and the
        filename utilizes the parameters from training and time of training to create the model name'''

    def save_model(self, tag=''):
        model_name = f'{self.date_t}--{tag}--{self.epochs_n}Epochs--{self.loss_t}-Loss--{self.hidden_units}-HU--{self.tag_t}'
        self.model.save(self.models_path + f'/{model_name}')

    '''After parsing the .csv and .json data and then converting into data accessible by Tensorflow, it can be pickled
        and reloaded next time a model needs to be trained. On Google Colab specifically, this can save ~10 minutes 
        from a session restart'''

    def save_data_and_labels(self):
        obj = {'XX_tf': self.XX_tf, 'YY_tf': self.YY_tf, 'entities': self.entities, 'entities_idx': self.entities_idx,
               'relations_dict': self.relations_dict,
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
