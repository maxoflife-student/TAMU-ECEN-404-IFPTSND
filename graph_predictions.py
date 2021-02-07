import os
import sys
from os.path import isfile, join

from tensorflow_models import leaky_relu, Ein_Multiply

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import random

from IPython.display import display
from bqplot import (
    LinearScale, Lines, Axis, Figure, Toolbar
)

import ipywidgets as widgets

import pickle
import json
import warnings
import os
from sklearn.metrics import mean_squared_error

import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Given a list, return the max index of that list
# If given rank, then it returns the rank-th highest index
def max_index(l, rank=0, mini=False):
    items = []
    for idx, value in enumerate(l):
        items.append((idx, value))

    # Sort the ratios and their indexes from high to low
    items.sort(key=lambda x: x[1], reverse=(not mini))

    return items[rank][0]


class Graph_Predictions():
    def __init__(self, model_path, results_path, tensorflow_model_obj):
        # Data that should be given to operate
        self.model_path = model_path
        self.results_path = results_path
        self.entities = tensorflow_model_obj.entities
        self.x_test = tensorflow_model_obj.data_splits['x_test']
        self.x_val = tensorflow_model_obj.data_splits['x_val']
        self.rr_test = tensorflow_model_obj.data_splits['rr_test']
        self.Normalized_Adjacency_Matrix = tensorflow_model_obj.Normalized_Adjacency_Matrix

        # Data used in our specific strategy implementation method
        self.increment = 5e4
        self.starting_investment = 2 * self.increment
        self.num_entities = self.x_test.shape[0]
        self.num_time_steps = self.x_test.shape[1]

        # To calculate MSE, YY_tf is needed from the tensorflow model
        self.YY_tf = tensorflow_model_obj.YY_tf

        # Daily return ratios have already been calculated, why redo them?
        self.RR_tf = tensorflow_model_obj.RR_tf

        # Data saved in memory
        self.strategy_results = {}

        # Load the files already in ./strategies
        self.update_strats()

    '''Loads all saved strategy results from the directory into memory as a dictionary'''

    def update_strats(self):
        # Load in the strategy names stored in
        p_files = self.generate_p_files()
        values = self.values_from_p(p_files)
        # Combine them into a dictionary
        self.strat_dict = {p_files[i]: values[i] for i in range(len(values))}

    '''Returns the list of pickled files in the directory and also calculates the average strategy if it has not been
        completed'''

    def generate_p_files(self):
        files = [f for f in os.listdir(self.results_path) if isfile(join(self.results_path, f))]
        files = [f for f in files if f.endswith('.p')]
        files.sort()

        # If the average strategy result hasn't been calculated yet, then calculate it and save it
        if '000_Average_000.p' not in files:
            self.strategy_average()
            self.save_results()

        return files

    '''Takes in a list of files, returns a list of list of their values'''

    def values_from_p(self, file):
        file_l = []
        for f in file:
            with open(self.results_path + '\\' + f, 'rb') as read:
                file_l.append(pickle.load(read))
        return file_l

    '''Displays the main graph for reviewing strategies in the Jupyter Notebook output window'''

    def strat_graph(self, strat_sel):

        # Load the files already in ./strategies
        self.update_strats()

        # Create a class variable for the selected feature
        self.strat_sel = strat_sel

        # Create some colors for the graphed lines to cycle through
        colors_list = ['0000FF', 'f38c00', 'FDD20E', '72e400', '00e4c1', '0076f3', '9700f3', 'f300c3', '990000',
                       'd08800', '05aa00', '00a0aa', '7600aa', '830000', 'f38c00', '995900', '7e6701', '264d00',
                       '004d41', '003166', '800066']
        colors_list = [f'#{i}' for i in colors_list]
        tmp = list(colors_list)
        for i in range(3):
            for color in tmp:
                colors_list.append(color)

        # Create distinctions between the colors
        line_styles = []
        # Account for the added one black at the end
        n = len(colors_list) - 1
        n4 = int(n / 4)
        for i in range(n):
            if i <= n4:
                line_styles.append('solid')
            elif n4 < i <= 2 * n4:
                line_styles.append('dashed')
            elif 2 * n4 < i <= 3 * n4:
                line_styles.append('dotted')
            else:
                line_styles.append('dash_dotted')

        # Insert the black color used on the selected entity
        colors_list.insert(0, '#f30000')

        # Insert the black color used on the selected entity
        colors_list.insert(0, '#008000')

        # Insert the black color used on the selected entity
        colors_list.insert(0, '#000000')

        self.max_l = 0
        for key in strat_sel:
            value = len(self.strat_dict[key])
            if value > self.max_l:
                self.max_l = value

        x_data = list(range(self.max_l))
        y_data = [self.strat_dict[key] for key in strat_sel]

        x_scale = LinearScale()
        y_scale = LinearScale()

        ax_x = Axis(scale=x_scale, label='Trading Days', grid_lines='solid')
        ax_y = Axis(scale=y_scale, label='Portfolio Total (USD)', orientation='vertical', label_offset='50px', )

        line = [Lines(labels=[strat_sel[i]], x=x_data, y=y_data[i], scales={'x': x_scale, 'y': y_scale},
                      colors=[colors_list[i]], display_legend=True, line_styles=line_styles[i]) for i in
                range(1, len(strat_sel))]
        # Line settings for the first entity
        line.insert(0, Lines(labels=[strat_sel[0]], x=x_data, y=y_data[0], scales={'x': x_scale, 'y': y_scale},
                             colors=[colors_list[0]], display_legend=True, stroke_width=2))

        fig = Figure(marks=line, axes=[ax_x, ax_y], title='Comparing Trading Strategies Over the Same Test Set',
                     colors=['red'], legend_location='top-left', legend_text={'font-size': 18})
        fig.layout.height = '950px'

        toolbar = Toolbar(figure=fig)

        display(fig, toolbar)

    def display_graph(self):

        # Update the list of available strategies
        self.update_strats()

        strat_sel = widgets.SelectMultiple(
            value=['000_Average_000.p'],
            options=self.strat_dict.keys(),
            rows=10,
            description='Strategies',
            disabled=False,
            layout=widgets.Layout(width='50%')
        )

        widgets.interact(self.strat_graph, strat_sel=strat_sel)

    '''Save the calculated results to a pickle file to be used later'''

    def save_results(self):
        for key, value in self.strategy_results.items():
            pickle.dump(value, open(self.results_path + fr'/{key}.p', 'wb'))
        self.update_strats()

    '''Given a company, day, and amount, returns the amount of money earned from buying it and then selling it
        the next day'''

    def buy_then_sell(self, company, day, amount):
        today_price = self.x_test[company, day, 0]
        tomorrow_price = self.x_test[company, day + 1, 0]
        return amount * (tomorrow_price / today_price)

    '''This strategy purchases a third of the entities each day at random and splits the total among them. This
        creates an average that can be compared against'''

    def strategy_average(self):
        total = self.starting_investment
        yesterday_earning = 0
        total_by_day = []

        divisor = int(len(self.entities) / 3)
        for day in range(1, self.num_time_steps - 1):

            # Choose 1/4 of the entities at random
            c_choices = [random.randint(0, len(self.entities) - 1) for i in range(divisor)]

            # Gain the money from yesterday's purchase
            total += yesterday_earning

            # Lose the money from the total that is spent today
            total -= self.increment

            # Calculate the amount of money that will be earned tomorrow when sold
            sum = 0
            for c in c_choices:
                sum = sum + self.buy_then_sell(c, day, self.increment / divisor)

            yesterday_earning = sum

            total_by_day.append(total)

        self.strategy_results['000_Average_000'] = total_by_day

    '''This strategy utilizes an LSTM model to generate a column vector containing every entity in the dataset. It
        chooses the entity that has the largest value in that column, signifying that the model predicts that stock
        to grow the most by the next day. This process repeats during the testing set'''

    def strategy_ratio_lstm(self, model_name, name_override='', avoid_fall=True, average=1, expVis=True):

        print(f"\nLoading Model: '{model_name}'")
        model = tf.keras.models.load_model(self.model_path + f'/{model_name}', compile=False,
                                           custom_objects={'leaky_relu': leaky_relu})

        if name_override:
            model_name = name_override

        # Portfolio to start
        total = self.starting_investment
        yesterday_earning = 0
        total_by_day = []

        # Only used if avoid_fall strategy
        losing_streak = -1

        for day in range(1, self.num_time_steps - 1):

            # Add some feedback into the post-prediction algorithm
            if avoid_fall:
                if yesterday_earning < self.increment:
                    losing_streak += 1

            # If bankrupt, stop the strategy
            if total < 0:
                break

            # The model should only be able to see up to yesterday
            seeable = self.x_test[:, 0:day, :]

            # Allow the model to see the validation set when predicting
            # ~Triples prediction time
            if expVis:
                seeable = tf.concat([self.x_val, seeable], axis=1)

            # Make a prediction
            pred = model.predict(tf.constant(seeable))

            # If money was lost on the last decision, choose the next best options(s)
            if avoid_fall:
                if yesterday_earning > self.increment:
                    c_choices = [max_index(pred, i) for i in range(average)]
                    losing_streak = 0
                else:
                    c_choices = [max_index(pred, i + losing_streak) for i in range(average)]

            else:
                c_choices = [max_index(pred, i) for i in range(average)]

            # Print out the day, how much money has currently been earned, and then which stock is about to be
            # purchased
            if day > 1:
                print(f"Day: {day - 1}\t\tTotal: {int(total)} Buying: {[self.entities[c] for c in c_choices]}")

            # Earn yesterday's money
            total += yesterday_earning
            # Lose the money from the total that is spent today
            total -= self.increment
            # Calculate the amount of money that will be earned tomorrow when sold
            sum = 0
            for c in c_choices:
                sum = sum + self.buy_then_sell(c, day, self.increment / len(c_choices))
            yesterday_earning = sum

            total_by_day.append(total)

        self.strategy_results[model_name] = total_by_day
        # Save the current strategy results
        self.save_results()

    '''This strategy utilizes an GCN model to generate a column vector containing every entity in the dataset. It
        chooses the entity that has the largest value in that column, signifying that the model predicts that stock
        to grow the most by the next day. This process repeats during the testing set'''

    def strategy_ratio_gcn(self, model_name, name_override='', avoid_fall=True, average=1, expVis=True):

        print(f"\nLoading Model: '{model_name}'")
        model = tf.keras.models.load_model(self.model_path + f'/{model_name}', compile=False,
                                           custom_objects={'Ein_Multiply': Ein_Multiply, 'leaky_relu': leaky_relu})

        if name_override:
            model_name = name_override

        # Portfolio to start
        total = self.starting_investment
        yesterday_earning = 0
        total_by_day = []

        # Only used if avoid_fall strategy
        losing_streak = -1

        for day in range(1, self.num_time_steps - 1):

            # Add some feedback into the post-prediction algorithm
            if avoid_fall:
                if yesterday_earning < self.increment:
                    losing_streak += 1

            # If bankrupt, stop the strategy
            if total < 0:
                break

            # The model should only be able to see up to yesterday
            seeable = self.x_test[:, 0:day, :]

            # Allow the model to see the validation set when predicting
            # ~Triples prediction time
            if expVis:
                seeable = tf.concat([self.x_val, seeable], axis=1)

            # Make a prediction
            pred = model.predict([tf.constant(seeable), self.Normalized_Adjacency_Matrix])

            # If money was lost on the last decision, choose the next best options(s)
            if avoid_fall:
                if yesterday_earning > self.increment:
                    c_choices = [max_index(pred, i) for i in range(average)]
                    losing_streak = 0
                else:
                    c_choices = [max_index(pred, i + losing_streak) for i in range(average)]

            else:
                c_choices = [max_index(pred, i) for i in range(average)]

            # Print out the day, how much money has currently been earned, and then which stock is about to be
            # purchased
            if day > 1:
                print(f"Day: {day - 1}\t\tTotal: {int(total)} Buying: {[self.entities[c] for c in c_choices]}")

            # Earn yesterday's money
            total += yesterday_earning
            # Lose the money from the total that is spent today
            total -= self.increment
            # Calculate the amount of money that will be earned tomorrow when sold
            sum = 0
            for c in c_choices:
                sum = sum + self.buy_then_sell(c, day, self.increment / len(c_choices))
            yesterday_earning = sum

            total_by_day.append(total)

        self.strategy_results[model_name] = total_by_day
        # Save the current strategy results
        self.save_results()

    '''Generates a json file containing every entity and its ratio prediction for all time-steps t in testing period'''

    def generate_prediction_json(self, model_name, name_override='', top_n=25, bottom_n=25, expVis=True,
                                 neural_net_type='lstm'):

        print(f"\nLoading Model: '{model_name}'")

        # Specify which parameter is being used to determine custom objects
        if neural_net_type == 'lstm':
            model = tf.keras.models.load_model(self.model_path + f'/{model_name}', compile=False,
                                               custom_objects={'leaky_relu': leaky_relu})
        elif neural_net_type == 'gcn':
            model = tf.keras.models.load_model(self.model_path + f'/{model_name}', compile=False,
                                               custom_objects={'Ein_Multiply': Ein_Multiply, 'leaky_relu': leaky_relu})
        else:
            input('The model type you specified was not found, so custom_objects cannot be applied. Fix this.')
            sys.exit()

        if name_override:
            model_name = name_override

        '''Deprecated, see below'''
        # top_entities = []
        # bottom_entities = []
        # mse = []

        # Create a dictionary with entity keys where all entities are an empty list
        results = {}
        for c in self.entities:
            results[c] = []

        print(f"Total number of days: {self.x_test.shape[1]}")

        for day in range(0, self.num_time_steps - 1):

            # On the first day, the model should only see the validation set
            # after that day, then the test set needs to be incrementally added in
            if day > 0:
                # The model should only be able to see up to yesterday
                seeable = self.x_test[:, 0:day, :]

                # Allow the model to see the validation set when predicting
                # ~Triples prediction time
                if expVis:
                    seeable = tf.concat([self.x_val, seeable], axis=1)
            else:
                seeable = self.x_val

            # Make a prediction, the input changes depending on the model type
            if neural_net_type == 'lstm':
                pred = model.predict(tf.constant(seeable))
            elif neural_net_type == 'gcn':
                pred = model.predict([tf.constant(seeable), self.Normalized_Adjacency_Matrix])

            # For all N, add its predictions to its containing list in the results dictionary
            for i, c in enumerate(self.entities):
                results[c].append(float(pred[i]))

            '''Was used to calculate the bottom & top companies, but is deprecated now
                that I'm just saving all prediction results'''
            # # Find the top n and bottom n entities
            # top_entities_c = [max_index(pred, i) for i in range(top_n)]
            # bottom_entities_c = [max_index(pred, i, mini=True) for i in range(bottom_n)]
            # # Convert their position to entity name for read-ability
            # top_entities_t = [self.entities[c] for c in top_entities_c]
            # bottom_entities_t = [self.entities[c] for c in bottom_entities_c]
            #
            # # Calculate the average MSE for company predictions
            # mse_t = np.mean(tf.keras.losses.mse(pred, self.YY_tf[:, day]))
            #
            # # Append the calculated values
            # top_entities.append(top_entities_t)
            # bottom_entities.append(bottom_entities_t)
            # mse.append(mse_t)

            print(f"Day {day} |", end=' ')

        '''Deprecated with previous calculations'''
        # results = {
        #     "top": top_entities,
        #     "bottom": bottom_entities,
        #     "mse": mse
        # }

        # This will be useful for making retrieving the predictions later
        # If x_test is always assumed to be at the tail of each dataset, then
        results['x_test_shape'] = list(self.x_test.shape)

        with open(f'./prediction_results/{model_name}_PM.json', 'w') as file:
            json.dump(results, file, indent=1)

    # This variation of the strategy execution assumes that the highest values in the prediction list are the best
    # stock.
    def prediction_json_strategy_max_entities(self, pm_name, name_override='', avoid_fall=True, average=1):

        print(f"\nLoading PM Model: '{pm_name}'")

        # Load in the prediction results as a dictionary
        file = open(f'./prediction_results/{pm_name}.json', 'r')
        pm = json.load(file)

        # Create a list of lists with shape (N, t)
        pred = [pm[c] for c in self.entities]
        # Transpose the list so we can index it as (t, N)
        pred = np.array(pred)
        pred_list = np.transpose(pred)

        if name_override:
            pm_name = name_override

        # Portfolio to start
        total = self.starting_investment
        yesterday_earning = 0
        total_by_day = []

        # Only used if avoid_fall strategy
        losing_streak = -1

        for day in range(0, self.num_time_steps - 2):

            # Add some feedback into the post-prediction algorithm
            if avoid_fall:
                if yesterday_earning < self.increment:
                    losing_streak += 1

            # If bankrupt, stop the strategy
            if total < 0:
                break

            # Make a prediction
            pred = list(pred_list[day])

            # If money was lost on the last decision, choose the next best options(s)
            if avoid_fall:
                if yesterday_earning > self.increment:
                    c_choices = [max_index(pred, i) for i in range(average)]
                    losing_streak = 0
                else:
                    c_choices = [max_index(pred, i + losing_streak) for i in range(average)]

            else:
                c_choices = [max_index(pred, i) for i in range(average)]

            # Print out the day, how much money has currently been earned, and then which stock is about to be
            # purchased
            if day > 0:
                print(f"Day: {day}\t\tTotal: {int(total)} Buying: {[self.entities[c] for c in c_choices]}")

            # Earn yesterday's money
            total += yesterday_earning
            # Lose the money from the total that is spent today
            total -= self.increment
            # Calculate the amount of money that will be earned tomorrow when sold
            sum = 0
            for c in c_choices:
                sum = sum + self.buy_then_sell(c, day, self.increment / len(c_choices))
            yesterday_earning = sum

            total_by_day.append(total)

        self.strategy_results[pm_name] = total_by_day
        # Save the current strategy results
        self.save_results()

    # This variation of the strategy execution assumes that each value is simply a prediction of the next days price,
    # so the highest ranked choices for the day need to be calculated from the closing price
    def prediction_json_strategy_determine_best(self, pm_name, name_override='', avoid_fall=True, average=1):

        print(f"\nLoading PM Model: '{pm_name}'")

        # Load in the prediction results as a dictionary
        file = open(f'./prediction_results/{pm_name}.json', 'r')
        pm = json.load(file)

        # Create a list of lists with shape (N, t)
        pred = [pm[c] for c in self.entities]
        # Transpose the list so we can index it as (t, N)
        pred = np.array(pred)
        pred_list = np.transpose(pred)

        if name_override:
            pm_name = name_override

        # Portfolio to start
        total = self.starting_investment
        yesterday_earning = 0
        total_by_day = []

        # Only used if avoid_fall strategy
        losing_streak = -1

        for day in range(1, self.num_time_steps - 2):

            # Add some feedback into the post-prediction algorithm
            if avoid_fall:
                if yesterday_earning < self.increment:
                    losing_streak += 1

            # If bankrupt, stop the strategy
            if total < 0:
                break

            # Make a prediction
            pred = pred_list[day]

            # Convert those predictions into highest difference
            actual = self.x_test[:, day - 1, 0]
            pred = tf.divide(tf.subtract(pred, actual), pred)
            pred = list(pred)

            # If money was lost on the last decision, choose the next best options(s)
            if avoid_fall:
                if yesterday_earning > self.increment:
                    c_choices = [max_index(pred, i) for i in range(average)]
                    losing_streak = 0
                else:
                    c_choices = [max_index(pred, i + losing_streak) for i in range(average)]

            else:
                c_choices = [max_index(pred, i) for i in range(average)]

            # Print out the day, how much money has currently been earned, and then which stock is about to be
            # purchased
            if day > 0:
                print(f"Day: {day}\t\tTotal: {int(total)} Buying: {[self.entities[c] for c in c_choices]}")

            # Earn yesterday's money
            total += yesterday_earning
            # Lose the money from the total that is spent today
            total -= self.increment
            # Calculate the amount of money that will be earned tomorrow when sold
            sum = 0
            for c in c_choices:
                sum = sum + self.buy_then_sell(c, day, self.increment / len(c_choices))
            yesterday_earning = sum

            total_by_day.append(total)

        self.strategy_results[pm_name] = total_by_day
        # Save the current strategy results
        self.save_results()

    def prediction_json_mse(self, pm_name, name_override=''):

        print(f"\nLoading PM Model: '{pm_name}'")

        # Load in the prediction results as a dictionary
        file = open(f'./prediction_results/{pm_name}.json', 'r')
        pm = json.load(file)

        # Create a list of lists with shape (N, t)
        pred = [pm[c] for c in self.entities]
        # Transpose the list so we can index it as (t, N)
        pred = np.array(pred)
        pred_list = np.transpose(pred)

        if name_override:
            pm_name = name_override

        daily_mse = []
        print(f"Num of time steps: {self.num_time_steps} | YY_tf: {self.YY_tf.shape}")
        for day in range(0, self.num_time_steps - 2):
            # Make a prediction
            pred = list(pred_list[day])

            # Compare it to the actual return ratios
            daily_mse.append(tf.keras.losses.mse(self.YY_tf[:, day], pred))

        self.strategy_results['MSE_Calc_' + pm_name] = daily_mse
        # Save the current strategy results
        self.save_results()

    def generate_upper_lower_avg_bounds(self):

        daily_highest_rr = []
        for day in range(self.num_time_steps):
            daily_highest_rr.append(np.amax(self.rr_test[:, day]))

        self.strategy_results['000_Highest_RR_Possible'] = daily_highest_rr

        daily_lowest_rr = []
        for day in range(self.num_time_steps):
            daily_lowest_rr.append(np.amin(self.rr_test[:, day]))

        self.strategy_results['000_Lowest_RR_Possible'] = daily_lowest_rr

        daily_avg_rr = []
        for day in range(self.num_time_steps):
            daily_avg_rr.append(np.mean(self.rr_test[:, day]))

        self.strategy_results['000_Avg_RR'] = daily_avg_rr
        self.save_results()

    # This variation of the strategy execution assumes that each value is simply a prediction of the next days price,
    # so the highest ranked choices for the day need to be calculated from the closing price

    # Saves the real RR given our predictions, the MSE between predicted RR and true RR, the Mean-Reciprocal-Rank
    # and the stock choice for the day
    def generate_model_diagnostics(self, pm_name, name_override='', datablock_folder='RL_validation_set'):

        # If these values haven't been calculated on this testing set yet, calculate them
        try:
            daily_lowest_rr = self.strat_dict['000_Lowest_RR_Possible.p']
            daily_highest_rr = self.strat_dict['000_Highest_RR_Possible.p']
            daily_avg_rr = self.strat_dict['000_Avg_RR.p']
        except KeyError:
            self.generate_upper_lower_avg_bounds()

        # Load in the prediction results as a dictionary
        file = open(f'./prediction_results/{pm_name}.json', 'r')
        pm = json.load(file)

        # Create a list of lists with shape (N, t)
        pred = [pm[c] for c in self.entities]
        # Transpose the list so we can index it as (t, N)
        pred = np.array(pred)
        pred_list = np.transpose(pred)

        if name_override:
            pm_name = name_override

        # Saves all RR values
        rr_list = []
        # Saves the company choices at each day
        entity_choices_list = []
        # Saves the MSE
        mse_list = []
        # Saves the Rank-Loss
        mrr_list = []

        # print("Day: ", end='')
        for day in range(0, self.num_time_steps - 2):
            # print(f"{day}", end='|')
            # Make a prediction for the price on day 0
            # Which means only the testing set has been seen
            pred = pred_list[day]

            if day == 0:
                yesterday_price = self.x_val[:, -1, 0]
            else:
                yesterday_price = self.x_test[:, day - 1, 0]

            # Convert those predictions into highest return_ratio
            pred = tf.divide(tf.subtract(pred, yesterday_price), pred)
            top_choice = np.argmax(pred)
            entity_choices_list.append(int(top_choice))

            choice_return_ratio = (self.x_test[top_choice, day + 1, 0] - self.x_test[top_choice, day, 0]) / self.x_test[
                top_choice, day + 1, 0]
            rr_list.append(float(choice_return_ratio))

            # Get the MSE
            mse_list.append(float(mean_squared_error(pred, self.rr_test[:, day])))

            # Calculate the MRR
            # Create lists for the predicted return ratios and actual return ratios
            predictions = pred.numpy()
            return_ratios = self.rr_test[:, day].numpy()

            # This algorithm is very fast for accurate models, possibly slow for inaccurate models
            # Iteritively returns the highest return_ratio in the prediction set. If it's not the same
            # as the ACTUAL highest return_ratio, then it lowers this value past the previous minimum and
            # searches for the next maximum. It repeats this until it determines the rank position
            actual_top = np.argmax(return_ratios)
            actual_bottom = np.argmin(predictions)
            count = 1
            for i in range(len(predictions)):
                inner_max = np.argmax(predictions)
                if inner_max == actual_top:
                    break
                else:
                    count += 1
                    predictions[inner_max] = predictions[actual_bottom] - 1
            mrr_list.append(float(1 / count))

            # This method will be slow regardless, but MIGHT be better on a totally random model
            # # Add 0, 1, 2, 3... to each value to remember each companies return ratio
            # predictions = list(zip(range(len(predictions)), predictions))
            # return_ratios = list(zip(range(len(return_ratios)), return_ratios))
            #
            # # Sort the list high to low for the return ratios
            # predictions.sort(key=lambda x: x[1], reverse=True)
            # return_ratios.sort(key=lambda x: x[1], reverse=True)
            #
            # # Traverse the predictions list until you reach the best choice.
            # actual_best_choice_index = return_ratios[0][0]
            # count = 1
            # for i in predictions:
            #     if i[0] == actual_best_choice_index:
            #         break
            #     else:
            #         count += 1
            # mrr_list.append(1 / count)

        # If you followed the predictions exactly each day, what is the overall return ratio?
        rr_list_plus_1 = (np.array(rr_list) + 1)
        cumulative_rr = float(np.prod(rr_list_plus_1))

        # What percentage of the stocks did the algorithm actually use? Is it always picking the same one?
        diversity_perc = float(len(set(entity_choices_list)) / self.x_test.shape[1])

        # What is the average error in calculating the next day return ratio?
        avg_mse = float(np.mean(mse_list))

        # What is the average reciprocal rank score? i.e. Where do we tend to rank the best choice?
        avg_mrr = float(np.mean(mrr_list))

        # Using area under the curve, what percentage of perfect did this model accomplish?
        test_set_low = np.mean(self.strat_dict['000_Lowest_RR_Possible.p'])
        test_set_high = np.mean(self.strat_dict['000_Highest_RR_Possible.p'])
        test_set_average = np.mean(self.strat_dict['000_Avg_RR.p'])

        avg_rr = np.mean(rr_list)
        if avg_rr > 0:
            best_potential_score = float((avg_rr - test_set_average) / (test_set_high - test_set_average))
        else:
            best_potential_score = float((avg_rr - test_set_average) / test_set_low - test_set_average) * -1

        print(test_set_low, test_set_high, test_set_average, avg_rr)

        json_file_save = {
            "Average_MRR": avg_mrr, "Average_MSE": avg_mse, "Diversity_Percentage": diversity_perc,
            "Cumulative_Return_Ratio": cumulative_rr, "MRR_List": mrr_list, "MSE_List": mse_list,
            "Entity_Choices": entity_choices_list, "Return_Ratio_List": rr_list, "Best_Potential_Score": best_potential_score
        }

        # Add a folder to store the results
        try:
            os.mkdir(f'./{datablock_folder}')
        except:
            None

        with open(f'./{datablock_folder}/{pm_name}_DATABLOCK.json', 'w') as file:
            json.dump(json_file_save, file, indent=1)

        self.strategy_results['RR_' + pm_name] = rr_list
        # Save the current strategy results
        self.save_results()
