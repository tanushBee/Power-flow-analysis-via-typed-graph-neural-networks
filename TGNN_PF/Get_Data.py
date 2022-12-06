import random
import sys

import numpy as np
import pandas as pd
from numpy.random import default_rng
import tensorflow as tf

# choose from data files
from Case30Net import *

from misc import v_check_control

class DataImporter(object):

    def __init__(self,
                 batch_size=10):

        self.batch_size = batch_size
        self.input_data = ref_grid()

        bus_i = np.reshape(self.input_data['bus'][:, 0], [1, -1])  # horizontallly
        fbus = np.reshape(self.input_data['branch'][:, 0], [-1, 1])  # vertically
        tbus = np.reshape(self.input_data['branch'][:, 1], [-1, 1])  # vertically
        gbus = np.reshape(self.input_data['gen'][:, 0], [-1, 1])  # vertically

        # start indices with 0
        self.fbus = np.where(bus_i - fbus == 0)[1]
        self.tbus = np.where(bus_i - tbus == 0)[1]
        self.gbus = np.where(bus_i - gbus == 0)[1]

        self.n_nodes = tf.Variable(self.input_data['bus'].shape[0], trainable=False, dtype=tf.int32)
        self.n_gens = tf.Variable(self.input_data['gen'].shape[0], trainable=False, dtype=tf.int32)
        self.baseMVA = self.input_data['baseMVA'] * np.ones([1, 1])

        # bus data
        self.buses_default = {}
        self.buses_default['Pd'] = tf.Variable(np.reshape(self.input_data['bus'][:, 2], [1, -1]), trainable=False,
                                               dtype=tf.float32)
        self.buses_default['Qd'] = tf.Variable(np.reshape(self.input_data['bus'][:, 3], [1, -1]), trainable=False,
                                               dtype=tf.float32)
        self.buses_default['Gs'] = tf.Variable(np.reshape(self.input_data['bus'][:, 4], [1, -1]), trainable=False,
                                               dtype=tf.float32)
        self.buses_default['Bs'] = tf.Variable(np.reshape(self.input_data['bus'][:, 5], [1, -1]), trainable=False,
                                               dtype=tf.float32)

        # lines data
        self.lines_default = {}
        # ratio: fill zeros with 1s.
        ratios = self.input_data['branch'][:, 8] + 1. * (self.input_data['branch'][:, 8] == 0.)
        self.lines_default['from'] = tf.Variable(np.reshape(self.fbus, [1, -1]), trainable=False,
                                                 dtype=tf.int32)
        self.lines_default['to'] = tf.Variable(np.reshape(self.tbus, [1, -1]), trainable=False,
                                               dtype=tf.int32)
        self.lines_default['r'] = tf.Variable(np.reshape(self.input_data['branch'][:, 2], [1, -1]), trainable=False,
                                              dtype=tf.float32)
        self.lines_default['x'] = tf.Variable(np.reshape(self.input_data['branch'][:, 3], [1, -1]), trainable=False,
                                              dtype=tf.float32)
        self.lines_default['b'] = tf.Variable(np.reshape(self.input_data['branch'][:, 4], [1, -1]), trainable=False,
                                              dtype=tf.float32)
        self.lines_default['ratio'] = tf.Variable(np.reshape(ratios, [1, -1]), trainable=False, dtype=tf.float32)

        # generators data
        self.gens_default = {}
        self.gens_default['bus'] = tf.Variable(np.reshape(self.gbus, [1, -1]), trainable=False,
                                               dtype=tf.int32)
        self.gens_default['Qg'] = tf.Variable(np.reshape(self.input_data['gen'][:, 2], [1, -1]), trainable=False,
                                              dtype=tf.float32)
        self.gens_default['Vg'] = tf.Variable(np.reshape(self.input_data['gen'][:, 5], [1, -1]), trainable=False,
                                              dtype=tf.float32)
        Pmax = tf.Variable(np.reshape(self.input_data['gen'][:, 8], [1, -1]), trainable=False,
                           dtype=tf.float32)
        Pmin = tf.Variable(np.reshape(self.input_data['gen'][:, 9], [1, -1]), trainable=False,
                                                dtype=tf.float32)
        self.gens_default['Pg_max_min'] = Pmax - Pmin
        self.gens_default['Pg_max'] = Pmax

        self.gens_default['mbase'] = tf.Variable(np.reshape(self.input_data['gen'][:, 6], [1, -1]), trainable=False,
                                                 dtype=tf.float32)

        # amount of samples
        self.n_samples = tf.Variable(self.batch_size, trainable=False)
        self.one_dim = tf.Variable(1, trainable=False)
        self.duplicator = tf.ones(tf.stack([self.n_samples, self.one_dim]), dtype=tf.float32)

        self.buses_batch = {}
        self.buses_batch_default = {}
        for key, value in self.buses_default.items():
            if key in ['Pd', 'Qd']:
                self.buses_batch_default[key] = self.buses_default[key] * self.duplicator
            # will NOT add noise to Gs, Bs:
            elif key in ['Gs', 'Bs']:
                self.buses_batch[key] = self.buses_default[key] * self.duplicator

        self.lines_batch = {}
        self.lines_batch_default = {}
        for key, value in self.lines_default.items():
            if key in ['from', 'to']:
                self.lines_batch[key] = self.lines_default[key] * tf.cast(self.duplicator, tf.int32)
            elif key in ['ratio']:
                self.lines_batch[key] = self.lines_default[key] * self.duplicator
            # WILL add noise to : R, X, B
            else:
                self.lines_batch_default[key] = self.lines_default[key] * self.duplicator

        self.gens_batch = {}
        self.gens_batch_default = {}
        for key, value in self.gens_default.items():
            if key in ['Pg_max_min', 'Vg']:
                self.gens_batch_default[key] = self.gens_default[key] * self.duplicator
            # will NOT add noise to:
            elif key in ['bus']:
                self.gens_batch[key] = self.gens_default[key] * tf.cast(self.duplicator, tf.int32)
            else:
                self.gens_batch[key] = self.gens_default[key] * self.duplicator

        self.linspace = tf.expand_dims(tf.range(0, self.n_samples, 1), -1)
        self.one_tensor = tf.ones([1], tf.int32)
        self.n_gens_tensor = tf.reshape(self.n_gens, [1])  # number of gens
        # concatenate vertically
        self.shape_gens_indices = tf.concat([self.one_tensor, self.n_gens_tensor], axis=0)
        # reshape horizontally
        self.indices_gens = tf.reshape(tf.tile(self.linspace, self.shape_gens_indices), [-1])
        self.indices_gens = tf.stack([self.indices_gens, tf.reshape(self.gens_batch['bus'], [-1])], 1) 
        self.indices_gens = tf.cast(self.indices_gens, tf.int64)

    def get_slack_info(self):

        self.slack_bus = np.squeeze(np.asarray(np.where(self.input_data['bus'][:, 1] == 3)))
        self.slack_bus_gens = np.squeeze(np.asarray(np.where(self.gbus == self.slack_bus)))

        return self.slack_bus, self.slack_bus_gens

    def get_batch(self,
                  sigma_inj=0.5,
                  sigma_vg=0.1,
                  sigma_pg=0.25,
                  sigma_lines=0.1):

        # Basically add noise and initialize voltage values.

        rng = default_rng()

        # little sanity check
        if self.buses_batch_default['Pd'].shape[0] != self.gens_batch_default['Pg_max_min'].shape[0]:
            print("SOMETHING IS WRONG WITH load/gens numer of samples")
        if self.buses_batch_default['Pd'].shape[0] != self.lines_batch_default['r'].shape[0]:
            print("SOMETHING IS WRONG WITH load/lines numer of samples")

        n_samples = self.buses_batch_default['Pd'].shape[0]

        # slack angle
        slack_ang = self.input_data['bus'][self.slack_bus, 8] * np.pi / 180
        slack_ang = tf.cast([slack_ang], tf.float32)

        # loads
        cols_bus = self.buses_batch_default['Pd'].shape[1]
        Pd_noise = rng.uniform(low=1-sigma_inj, high=1+sigma_inj, size=(n_samples, cols_bus))
        Qd_noise = rng.uniform(low=1-sigma_inj, high=1+sigma_inj, size=(n_samples, cols_bus))
        self.buses_batch['Pd'] = self.buses_batch_default['Pd'] * Pd_noise
        self.buses_batch['Qd'] = self.buses_batch_default['Qd'] * Qd_noise

        # generators
        cols = self.gens_batch_default['Vg'].shape[1]

        gen_noise = rng.uniform(low=sigma_pg, high=1-sigma_pf, size=(n_samples, cols))
        self.gens_batch['Pg'] = self.gens_batch_default['Pg_max_min'] * gen_noise

        Vg_noise = rng.uniform(low=1-sigma_vg, high=1+sigma_vg, size=(n_samples, cols))
        self.gens_batch['Vg'] = self.gens_batch_default['Vg'] * Vg_noise

        # lines
        cols = self.lines_batch_default['r'].shape[1]

        R_noise = rng.uniform(low=1-sigma_lines, high=1+sigma_lines, size=(n_samples, cols))
        X_noise = rng.uniform(low=1-sigma_lines, high=1+sigma_lines, size=(n_samples, cols))
        B_noise = rng.uniform(low=1-sigma_lines, high=1+sigma_lines, size=(n_samples, cols))

        self.lines_batch['r'] = self.lines_batch_default['r'] * R_noise
        self.lines_batch['x'] = self.lines_batch_default['x'] * X_noise
        self.lines_batch['b'] = self.lines_batch_default['b'] * B_noise

        # deleting random lines in each sample
        random_ints = rng.integers(low=1e-4, high=self.fbus.size, size=self.batch_size)
        all_lines = np.array(range(self.fbus.size))
        rand_lines = np.zeros([self.batch_size, self.fbus.size - 1])
        for i in range(self.batch_size):
            rand_lines[i, :] = np.delete(all_lines, random_ints[i])
        rand_indices = tf.constant(rand_lines, dtype=tf.int32)
        self.lines_reduced = {}
        for key, value in self.lines_batch.items():
            self.lines_reduced[key] = tf.gather(self.lines_batch[key], rand_indices, axis=1, batch_dims=1)

        # initial voltage
        self.V = {}
        # phase
        # default angle is slack angle:
        self.V['ang'] = np.zeros([n_samples, self.n_nodes.numpy()]) + slack_ang
        self.V['ang'] = tf.cast(self.V['ang'], tf.float32)

        # magnitude
        self.V['mag'] = tf.ones([self.batch_size, self.n_nodes.numpy()], dtype=tf.float32)
        self.V['mag'] = v_check_control(self.V['mag'], self.gens_batch['Vg'], self.indices_gens)

        return self.buses_batch, self.gens_batch, self.lines_reduced, self.V

