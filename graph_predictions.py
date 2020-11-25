import os
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
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Given a list, return the max index of that list
# If given rank, then it returns the rank-th highest index
def max_index(l, rank=0):
    items = []
    for idx, value in enumerate(l):
        items.append((idx, value))

    # Sort the ratios and their indexes from high to low
    items.sort(key=lambda x: x[1], reverse=True)

    return (items[rank][0])


class Graph_Predictions():
    def __init__(self, model_path, results_path, tensorflow_model_obj):
        # Data that should be given to operate
        self.model_path = model_path
        self.results_path = results_path
        self.entities = tensorflow_model_obj.entities
        self.x_test = tensorflow_model_obj.data_splits['x_test']
        self.x_val = tensorflow_model_obj.data_splits['x_val']
        self.Normalized_Adjacency_Matrix = tensorflow_model_obj.Normalized_Adjacency_Matrix

        # Data used in our specific strategy implementation method
        self.increment = 5e4
        self.starting_investment = 2 * self.increment
        self.num_entities = self.x_test.shape[0]
        self.num_time_steps = self.x_test.shape[1]

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
        if '0_Average_0.p' not in files:
            self.strategy_average()
            self.save_results(self.results_path)

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
        colors_list = ['f30000', 'f38c00', 'FDD20E', '72e400', '00e4c1', '0076f3', '9700f3', 'f300c3', '990000',
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
                             colors=[colors_list[0]], display_legend=True, stroke_width=6))

        fig = Figure(marks=line, axes=[ax_x, ax_y], title='Comparing Trading Strategies Over the Same Test Set',
                     colors=['red'], legend_location='top-left', legend_text={'font-size': 18})
        fig.layout.height = '950px'

        toolbar = Toolbar(figure=fig)

        display(fig, toolbar)

    def display_graph(self):

        # Update the list of available strategies
        self.update_strats()

        strat_sel = widgets.SelectMultiple(
            value=['0_Average_0.p'],
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

        self.strategy_results['0_Average_0'] = total_by_day

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
