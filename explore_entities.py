import sys

import os
from os import listdir
from os.path import isfile, join

import datetime
import time

from pathlib import Path

import IPython
import IPython.display
import numpy as np
import pandas as pd
import tensorflow as tf

import random
from random import shuffle

import numpy as np
from IPython.display import display
from bqplot import (
    LinearScale, Lines, Axis, Figure, Toolbar, ColorScale
)

import ipywidgets as widgets
from ipywidgets import Layout
import matplotlib.colors as colors

import pickle
import json

'''Given an item and a list, if that item is in the list, returns the index of that item'''


def return_idx(item, l):
    i = 0
    for thing in l:
        if item == thing:
            return i
        else:
            i += 1
    # If it's not in the list, then it's failed
    return None


'''Given a data_set folder with entities, provides tools to graph the entites'''


class Graph_Entities():
    def __init__(self, path):
        # The directory where all the entities time_series CSV are stored
        self.path = path
        self.entities, self.entities_idx = self._generate_list_of_entities()
        self.relations_dict = self._generate_relations()
        self.Normalized_Adjacency_Matrix = self._generate_normalized_ajacency_matrix()

    '''Using the names of .csv files in the data_set directory, loads the entities as a list into memory'''

    def _generate_list_of_entities(self):
        files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        ents = [i.replace('.csv', '') for i in files if i.endswith('.csv')]
        ents.sort()
        ents_idx = {item: idx for idx, item in enumerate(ents)}
        return ents, ents_idx

    '''Using the only .json file in the data_set directory, loads the relations as a dictionary into memory'''

    def _generate_relations(self):
        files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        relation_file = [i for i in files if i.endswith('.json')]

        if len(relation_file) == 1:
            # Load the relationship dictionary
            with open(self.path + '\\' + relation_file[0]) as read_file:
                relations_dict = json.load(read_file)
                return relations_dict

        elif len(relation_file) == 0:
            print('Directory does not contain an entity relationship .json file')

        else:
            sys.exit('There are multiple .json files in the directory, remove all or leave 1')

    '''Given a list of entities, return a list of their values averaged over the time-period'''
    '''Will be useful in detecting when a Neighboring Group moves NOT in accordance with the average'''

    def _generate_average_entity(self, entities):
        None

    '''Generates the normalized adjacency matrix from the relations dictionary'''

    def _generate_normalized_ajacency_matrix(self):
        companies = self.entities
        new_industry_relations = self.relations_dict

        # Iterate through each company ticker and replace it with a tuple that contains its index and ticker
        for key, value in new_industry_relations.items():
            new_value = []
            for v in value:
                new_value.append((return_idx(v, companies), v))
            new_industry_relations[key] = new_value

        # Iterate through each industry relationship and create an N x N adjacency matrix
        # Combine them all to create the final adjacency matrix in the same format as Paper #2
        RR_t = []
        for sector in new_industry_relations.keys():
            # Create an empty relationship matrix
            all_zeroes = tf.zeros([len(companies), len(companies)])
            relation_slice = all_zeroes.numpy()

            # Gather all the companies that exist in this sector
            siblings = new_industry_relations[sector]
            for i in siblings:
                for j in siblings:
                    relation_slice[i[0], j[0]] = 1
                    relation_slice[j[0], i[0]] = 1
            RR_t.append(relation_slice)

        RR_tf = tf.constant(RR_t)
        RR_tf = tf.transpose(RR_tf)
        relation_encoding = RR_tf.numpy()
        rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
        mask_flags = np.equal(np.zeros(rel_shape, dtype=int), np.sum(relation_encoding, axis=2))

        ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float), np.ones(rel_shape, dtype=float))

        degree = np.sum(ajacent, axis=0)
        for i in range(len(degree)):
            degree[i] = 1.0 / degree[i]
        np.sqrt(degree, degree)
        deg_neg_half_power = np.diag(degree)

        GCN_mat = np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)

        GCN_mat = np.nan_to_num(GCN_mat)
        GCN_mat = tf.constant(GCN_mat)
        return GCN_mat

    '''Returns a list of neighboring entities to the given an entity, includes the given entity'''

    def _return_neighbors(self, ent):
        # List containing the idx of all entities that are neighbors to the given entity
        neighbors = [self.entities[idx] for idx, item in
                     enumerate(self.Normalized_Adjacency_Matrix[self.entities_idx[ent]]) if item > 0]
        return neighbors

    '''Displays the explorable bqplot with Ipywidgets'''

    def _entity_graph(self, sel_feature, x_range, n_range, show_rel):
        # Create a class variable for the selected feature
        self.sel_feature = sel_feature
        self.x_range = x_range
        self.show_rel = show_rel
        self.n_range = n_range

        # Create some colors for the graphed lines to cycle through
        colors_list = ['f30000', 'f38c00', 'FDD20E', '72e400', '00e4c1', '0076f3', '9700f3', 'f300c3', '990000',
                       'd08800', '05aa00', '00a0aa', '7600aa', '830000', 'f38c00', '995900', '7e6701', '264d00',
                       '004d41', '003166', '800066']
        colors_list = [f'#{i}' for i in colors_list]
        tmp = list(colors_list)
        for i in range(4):
            for color in tmp:
                colors_list.append(color)

        # Insert the black color used on the selected entity
        colors_list.insert(0, '#000000')

        # Create distinctions between the colors
        line_styles = []
        # Account for the added one black at the end
        n = len(colors_list) - 1
        n4 = int(n/4)
        for i in range(n):
            if i < n4:
                line_styles.append('solid')
            elif n4 < i < 2*n4:
                line_styles.append('dashed')
            elif 2*n4 < i < 3*n4:
                line_styles.append('dotted')
            else:
                line_styles.append('dash_dotted')

        # There will always be one entity selected, so it sets the X-axis
        ent_df = pd.read_csv(self.path + '\\' + self.sel_ent + ".csv")
        x_data = list(range(len(ent_df[ent_df.columns[self.feature_key[self.sel_feature]]].values)))[
                 x_range[0]:x_range[1]]

        if not self.show_rel:
            y_data = ent_df[ent_df.columns[self.feature_key[self.sel_feature]]].values
        else:
            # Key names to be displayed
            keys = self._return_neighbors(self.sel_ent)
            # Remove the the main key we're looking for
            keys.remove(self.sel_ent)
            # Add it back to the front of the list so it's always displayed
            keys.insert(0, self.sel_ent)

            # List of DataFrames containing all the entities related to each other
            list_of_dfs = [pd.read_csv(self.path + '\\' + entity + ".csv") for entity in keys]
            # List of values from those DataFrames
            values = [df[df.columns[self.feature_key[self.sel_feature]]].values for df in list_of_dfs]

        x_scale = LinearScale()
        y_scale = LinearScale()

        ax_x = Axis(scale=x_scale, label='Time Steps', grid_lines='solid')
        ax_y = Axis(scale=y_scale, label='Value', orientation='vertical', label_offset='50px', )

        if not self.show_rel:
            line = [Lines(labels=[self.sel_ent], x=x_data, y=y_data, scales={'x': x_scale, 'y': y_scale},
                          display_legend=True)]
        else:
            n_range_of_entities = list(range(len(values)))[n_range[0]:n_range[1]]
            n_range_of_entities.insert(0, 0)

            # Line settings for all neighbors
            line = [Lines(labels=[keys[i]], x=x_data, y=values[i], scales={'x': x_scale, 'y': y_scale},
                          colors=[colors_list[i]], display_legend=True, line_style=line_styles[i]) for i in n_range_of_entities[1:]]

            # Line settings for the selected entity
            line.insert(0, Lines(labels=[keys[0]], x=x_data, y=values[0], scales={'x': x_scale, 'y': y_scale},
                                 colors=[colors_list[0]], display_legend=True, stroke_width=6))
            print('Test')

        fig = Figure(marks=line, axes=[ax_x, ax_y], title='Value of Entity(s) Over Time', colors=['red'],
                     legend_location='top-left')
        fig.layout.height = '850px'
        toolbar = Toolbar(figure=fig)
        display(fig, toolbar)

    '''This is the method to call to display the Ipywidgets and Graph'''

    def display_entities(self):
        # Create a dropdown menu to select which entity you would like to view
        ent_drop = widgets.Dropdown(
            options=self.entities,
            description='Entities: ',
            layout=Layout(justify_content='flex-start')
        )

        def _sel_feature(sel_ent):
            # Declare as a class variables
            self.sel_ent = sel_ent

            # Load in the data for the selected entity
            ent_df = pd.read_csv(self.path + '\\' + self.sel_ent + ".csv")
            # Create a dropdown menu to select which feature you would like to view
            ent_features = [f'{i}' for i in ent_df.columns]

            self.feature_key = {name: i for i, name in enumerate(ent_df.columns)}

            feat_drop = widgets.Dropdown(
                options=ent_features,
                description='Features: ',
                layout=Layout(justify_content='flex-end')
            )

            # Changes the default slider values to be the previous iteration if the main entity is changed
            try:
                start = self.x_range[0]
                end = self.x_range[1]
            except:
                start = 0
                end = 300

            x_range = widgets.IntRangeSlider(
                value=[start, end],
                min=0,
                max=ent_df.shape[0],
                step=10,
                description='Time Range:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d',
                layout=Layout(width='100%', align_items='stretch')
            )

            n_range = widgets.IntRangeSlider(
                value=[1, 1],
                min=1,
                max=len(self._return_neighbors(sel_ent)),
                step=1,
                description='Neighbors:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d',
                layout=Layout(width='40%', align_items='stretch')
            )

            show_rel = widgets.Checkbox(
                value=False,
                description='Show Neighbors',
                disabled=False,
            )

            widgets.interact(self._entity_graph, sel_feature=feat_drop, x_range=x_range, n_range=n_range,
                             show_rel=show_rel)

        widgets.interact(_sel_feature, sel_ent=ent_drop)
