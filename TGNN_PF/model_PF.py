import os
import json
import logging
import copy
import time

import tensorflow as tf
import numpy as np
import pandas as pd

import tgnv5_PF
from FuncDeltas import func_deltas
from misc import scatter_nd_add_diff, v_check_control, s_check_control
from Get_Data import DataImporter

from tqdm import tqdm

class TGN_PF(object):
    def __init__(self,
                 lr=1e-3,
                 batch_size=10,
                 dim_e=10,
                 dim_pv=10,
                 dim_pq=10,
                 time_steps=2,
                 tgn_layers=10,
                 non_lin='tanh',
                 name='TGN_PFsolverTop',
                 directory='./',
                 model_to_restore=None):

        # training parameters
        self.batch_size = batch_size
        self.lr = lr

        # model parameters
        self.in_dim = {"V_PQ": 4, "V_PV": 3}
        self.out_dim = {"V_PQ": 2, "V_PV": 1, "V_E": None, "V_S": None}

        self.dim_e = dim_e
        self.dim_pv = dim_pv
        self.dim_pq = dim_pq
        self.time_steps = time_steps
        self.tgn_layers = tgn_layers
        self.non_lin = non_lin
        self.name = name
        self.directory = directory
        self.scaling_factor = 1e-2

        self.current_train_iter = 0

        # Reload config if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):
            print("came in to model to restore")
            logging.info('    Restoring model from ' + model_to_restore)
            path_to_config = os.path.join(model_to_restore, 'config.json')
            with open(path_to_config, 'r') as f:
                config = json.load(f)
            self.set_config(config)

        # import default data
        self.data_importer = DataImporter(batch_size=self.batch_size)
        self.slack_bus, self.slack_bus_gens = self.data_importer.get_slack_info()

        # tgn
        var, mat, msg, loop = self.define_tgn(self.dim_e, self.dim_pv, self.dim_pq)
        self.tgn_model = tgnv5_pf.TGNmodel(var, mat, msg, loop, self.in_dim, self.out_dim, self.time_steps)

        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Restore trained weights if there is a model  restore
        self.checkpoint_prefix = os.path.join(self.directory, 'model.ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.tgn_model)
        self.checkpoint.restore(tf.train.latest_checkpoint(directory))

        # Log config infos
        self.log_config()

    def log_config(self):
        """Logs the config of the whole model"""

        logging.info('    Configuration: ')
        logging.info('    Latent E dimension: {}'.format(self.dim_e))
        logging.info('    Latent PV dimension: {}'.format(self.dim_pv))
        logging.info('    Latent PQ dimension: {}'.format(self.dim_pq))
        logging.info('    Message passing steps: {}'.format(self.time_steps))
        logging.info('    Total TGN layers: {}'.format(self.tgn_layers))
        logging.info('    Non-linearity: {}'.format(self.non_lin))
        logging.info('    Input dim: {}'.format(self.in_dim))
        logging.info('    Output dim: {}'.format(self.out_dim))
        logging.info('    Learning rate: {}'.format(self.lr))
        logging.info('    Batch size: {}'.format(self.batch_size))
        logging.info('    Current training iteration : {}'.format(self.current_train_iter))
        logging.info('    Model name : ' + self.name)

    def get_config(self):
        """
        Gets the config dict
        """
        config = {
            'latent_dimension_e': self.dim_e,
            'latent_dimension_pv': self.dim_pv,
            'latent_dimension_pq': self.dim_pq,
            'tgn_layers': self.tgn_layers,
            'time_steps': self.time_steps,
            'non_lin': self.non_lin,
            'input_dim': self.in_dim,
            'output_dim': self.out_dim,
            'learning_rate': self.lr,
            'batch_size': self.batch_size,
            'name': self.name,
            'directory': self.directory,
            'current_train_iter': self.current_train_iter
        }
        return config

    def set_config(self, config):
        """
        Sets the config according to an inputed dict
        """
        self.dim_e = config['latent_dimension_e']
        self.dim_pv = config['latent_dimension_pv']
        self.dim_pq = config['latent_dimension_pq']
        self.time_steps = config['time_steps']
        self.tgn_layers = config['tgn_layers']
        self.non_lin = config['non_lin']
        self.in_dim = config['input_dim']
        self.out_dim = config['output_dim']
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.name = config['name']
        self.directory = config['directory']
        self.current_train_iter = config['current_train_iter']

    def save(self):
        """
        Saves the configuration of the model and the trained weights
        """
        # save config
        config = self.get_config()
        path_to_config = os.path.join(self.directory, 'config.json')
        with open(path_to_config, 'w') as f:
            json.dump(config, f)

        # save weights
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def define_tgn(self, dim_e, dim_pv, dim_pq):

        var = {"V_E": dim_e, "V_PQ": dim_pq, "V_PV": dim_pv, "V_S": dim_pv}
        mat = {
            "M_PQ_E": ("V_PQ", "V_E"),
            "M_PV_E": ("V_PV", "V_E"),
            "M_S_E": ("V_S", "V_E")
        }
        msg = {
            "PQ_cast_E": ("V_PQ", "V_E"),
            "PV_cast_E": ("V_PV", "V_E"),
            "S_cast_E": ("V_S", "V_E"),
            "E_cast_PQ": ("V_E", "V_PQ"),
            "E_cast_PV": ("V_E", "V_PV"),
            "E_cast_S": ("V_E", "V_S")
        }
        loop = {
            "V_E": [
                {
                    "var": "V_E"
                },
                {
                    "mat": "M_PQ_E",
                    "transpose?": True,
                    "msg": "PQ_cast_E",
                    "var": "V_PQ"
                },
                {
                    "mat": "M_PV_E",
                    "transpose?": True,
                    "msg": "PV_cast_E",
                    "var": "V_PV"
                },
                {
                    "mat": "M_S_E",
                    "transpose?": True,
                    "msg": "S_cast_E",
                    "var": "V_S"
                }
            ],
            "V_PQ": [
                {
                    "var": "V_PQ"
                },
                {
                    "mat": "M_PQ_E",
                    "msg": "E_cast_PQ",
                    "var": "V_E"
                }
            ],
            "V_PV": [
                {
                    "var": "V_PV"
                },
                {
                    "mat": "M_PV_E",
                    "msg": "E_cast_PV",
                    "var": "V_E"
                }
            ],
            "V_S": [
                {
                    "var": "V_S"
                }
            ]
        }
        return var, mat, msg, loop

    def determine_type(self, n_buses, gen_ind):
        PV_buses = []
        PQ_buses = []
        S_buses = []
        PV_buses_local = []
        PQ_buses_local = []
        S_buses_local = []
        n_samples = gen_ind.shape[0]
        for sample in range(n_samples):
            for n in range(n_buses):
                if n in gen_ind[sample, :]:
                    if n != self.slack_bus:
                        PV_buses.append(n + sample*n_buses)
                        PV_buses_local.append(n)
                    else:
                        S_buses.append(n + sample * n_buses)
                        S_buses_local.append(n)
                else:
                    PQ_buses.append(n + sample*n_buses)
                    PQ_buses_local.append(n)

        PQ_buses = np.array(PQ_buses)
        PQ_buses = np.reshape(PQ_buses, [n_samples, -1])
        PV_buses = np.array(PV_buses)
        PV_buses = np.reshape(PV_buses, [n_samples, -1])
        S_buses = np.array(S_buses)
        S_buses = np.reshape(S_buses, [n_samples, -1])

        PQ_buses_local = np.array(PQ_buses_local)
        PQ_buses_local = np.reshape(PQ_buses_local, [n_samples, -1])
        PV_buses_local = np.array(PV_buses_local)
        PV_buses_local = np.reshape(PV_buses_local, [n_samples, -1])
        S_buses_local = np.array(S_buses_local)
        S_buses_local = np.reshape(S_buses_local, [n_samples, -1])

        return PV_buses, PV_buses_local, PQ_buses, PQ_buses_local, S_buses, S_buses_local

    def sparse_adj_mat(self, froms, tos, n_nodes, PVs, PQs, Ss):

        from_list = []
        to_list = []
        for i in range(self.batch_size):
            from_ind = froms[i, :] + (i * n_nodes)
            to_ind = tos[i, :] + (i * n_nodes)
            from_list.append(from_ind)
            to_list.append(to_ind)

        froms = tf.stack(from_list)
        froms = np.reshape(froms, [1, -1])

        tos = tf.stack(to_list)
        tos = np.reshape(tos, [1, -1])

        PV_ind = np.reshape(PVs, [1, -1])
        PQ_ind = np.reshape(PQs, [1, -1])
        S_ind = np.reshape(Ss, [1, -1])

        n_samples = PVs.shape[0]
        M_PV_E = np.zeros([PV_ind.size, froms.size])
        M_PQ_E = np.zeros([PQ_ind.size, froms.size])
        M_S_E = np.zeros([S_ind.size, froms.size])
        M_all_E = np.zeros([n_nodes * n_samples, froms.size])

        for i in range(froms.size):
            M_all_E[froms[:, i], i] = 1
            M_all_E[tos[:, i], i] = 1

        for i in range(PV_ind.size):
            M_PV_E[i, :] = M_all_E[int(PV_ind[:, i]), :]
        pv_line_ind = tf.convert_to_tensor(np.transpose(np.nonzero(M_PV_E)), dtype=tf.int64)
        pv_line_val = tf.ones([pv_line_ind.shape[0]])
        pv_line_shape = tf.constant([PV_ind.size, froms.size], dtype=tf.int64)
        sparse_PV_E = tf.sparse.SparseTensor(indices=pv_line_ind, values=pv_line_val, dense_shape=pv_line_shape)

        for i in range(PQ_ind.size):
            M_PQ_E[i, :] = M_all_E[int(PQ_ind[:, i]), :]
        pq_line_ind = tf.convert_to_tensor(np.transpose(np.nonzero(M_PQ_E)), dtype=tf.int64)
        pq_line_val = tf.ones([pq_line_ind.shape[0]])
        pq_line_shape = tf.constant([PQ_ind.size, froms.size], dtype=tf.int64)
        sparse_PQ_E = tf.sparse.SparseTensor(indices=pq_line_ind, values=pq_line_val, dense_shape=pq_line_shape)

        for i in range(S_ind.size):
            M_S_E[i, :] = M_all_E[int(S_ind[:, i]), :]
        s_line_ind = tf.convert_to_tensor(np.transpose(np.nonzero(M_S_E)), dtype=tf.int64)
        s_line_val = tf.ones([s_line_ind.shape[0]])
        s_line_shape = tf.constant([S_ind.size, froms.size], dtype=tf.int64)
        sparse_S_E = tf.sparse.SparseTensor(indices=s_line_ind, values=s_line_val, dense_shape=s_line_shape)

        return {"M_PV_E": sparse_PV_E,
                "M_PQ_E": sparse_PQ_E,
                "M_S_E": sparse_S_E}

    def get_tgn_input(self, buses_batch, gens_batch, lines_batch, V_batch, PV_buses, PQ_buses, S_buses):
        delta_p, delta_q, p_gens, q_gens = func_deltas(buses_batch, gens_batch, lines_batch, V_batch,
                                             self.slack_bus, self.slack_bus_gens)

        # PV bus nodes
        vmag_pv = tf.gather(V_batch['mag'], PV_buses, batch_dims=1)
        vang_pv = tf.gather(V_batch['ang'], PV_buses, batch_dims=1)
        dp_pv = tf.gather(delta_p, PV_buses, batch_dims=1)
        qg_pv = tf.gather(q_gens, PV_buses, batch_dims=1)

        # PQ bus nodes
        vmag_pq = tf.gather(V_batch['mag'], PQ_buses, batch_dims=1)
        vang_pq = tf.gather(V_batch['ang'], PQ_buses, batch_dims=1)
        dp_pq = tf.gather(delta_p, PQ_buses, batch_dims=1)
        dq_pq = tf.gather(delta_q, PQ_buses, batch_dims=1)

        # S bus nodes
        vmag_s = tf.gather(V_batch['mag'], S_buses, batch_dims=1)
        vang_s = tf.gather(V_batch['ang'], S_buses, batch_dims=1)
        pg_s = tf.gather(p_gens, S_buses, batch_dims=1)
        qg_s = tf.gather(q_gens, S_buses, batch_dims=1)

        # E nodes
        y_ij = 1. / tf.sqrt(lines_batch['r'] ** 2 + lines_batch['x'] ** 2)
        delta_ij = tf.math.atan2(lines_batch['r'], lines_batch['x'])

        pv_features = tf.stack([
            tf.reshape(vmag_pv, [-1]),
            tf.reshape(vang_pv, [-1]),
            tf.reshape(dp_pv, [-1])],
            axis=1)
        # print("pv feat: ", pv_features)

        pq_features = tf.stack([
            tf.reshape(vmag_pq, [-1]),
            tf.reshape(vang_pq, [-1]),
            tf.reshape(dp_pq, [-1]),
            tf.reshape(dq_pq, [-1])],
            axis=1)
        # print("pq feat: ", pq_features)
        s_features = tf.stack([
            tf.reshape(vmag_s, [-1]),
            tf.reshape(vang_s, [-1]),
            tf.reshape(pg_s, [-1]),
            tf.reshape(qg_s, [-1])],
            axis=1
        )

        e_features = tf.stack([
            tf.reshape(y_ij, [-1]),
            tf.reshape(delta_ij, [-1]),
            tf.reshape(lines_batch['b'], [-1])],
            axis=1)
        # print("e feat: ", e_features)

        return {"V_E": e_features,
                "V_PV": pv_features,
                "V_PQ": pq_features,
                "V_S": s_features}


    def update_V(self, deltaPV, deltaPQ, PV_nodes, PQ_nodes, V_batch):

        total_nodes = tf.constant([V_batch['mag'].shape[0] * V_batch['mag'].shape[1]], dtype=tf.int64)

        dPVang = tf.scatter_nd(tf.reshape(PV_nodes, [-1, 1]),
                               tf.squeeze(deltaPV * self.scaling_factor),
                               total_nodes)
        dPQang = tf.scatter_nd(tf.reshape(PQ_nodes, [-1, 1]),
                               tf.squeeze(tf.slice(deltaPQ * self.scaling_factor, [0, 0], [deltaPQ.shape[0], 1])),
                               total_nodes)
        dPQmag = tf.scatter_nd(tf.reshape(PQ_nodes, [-1, 1]),
                               tf.squeeze(tf.slice(deltaPQ * self.scaling_factor, [0, 1], [deltaPQ.shape[0], 1])),
                               total_nodes)

        delta_ang = tf.reshape(dPVang + dPQang, V_batch['mag'].shape)
        newVang = V_batch['ang'] + delta_ang

        delta_mag = tf.reshape(dPQmag, V_batch['mag'].shape)
        newVmag = V_batch['mag'] + delta_mag

        return newVmag, newVang

    def cost_func(self, dP, dQ):
        cost = tf.reduce_mean(2*tf.pow(dP, 2)+tf.pow(dQ, 2))
        return cost

    def train(self,
              max_iter,
              glob_norm,
              save_step=None):

        # Log infos about training process
        logging.info('    Starting a training process :')
        logging.info('    Max iteration : {}'.format(max_iter))
        logging.info('    Saving model every {} iterations'.format(save_step))

        starting_point = copy.copy(self.current_train_iter)

        buses_batch, gens_batch, lines_batch, V_batch = self.data_importer.get_batch()
        n_nodes = V_batch['mag'].shape[1]
        # types of nodes doesn't change
        PV_nodes, PV_nodes_l, PQ_nodes, PQ_nodes_l, S_nodes, S_nodes_l = self.determine_type(n_nodes, gens_batch['bus'])

        # training loop
        for i in tqdm(range(starting_point, starting_point + max_iter)):
            self.current_train_iter = i
            buses_batch, gens_batch, lines_batch, V_batch = self.data_importer.get_batch()
            adj_matrices = self.sparse_adj_mat(lines_batch['from'], lines_batch['to'], n_nodes,
                                             PV_nodes, PQ_nodes, S_nodes)

            with tf.GradientTape() as tape:
                for tgn_layer in range(self.tgn_layers):
                    nn_input = self.get_tgn_input(buses_batch, gens_batch, lines_batch, V_batch,
                                                  PV_nodes_l, PQ_nodes_l, S_nodes_l)
                    deltaPV, deltaPQ = self.tgn_model(adj_matrices, nn_input)
                    V_batch['mag'], V_batch['ang'] = self.update_V(deltaPV, deltaPQ, PV_nodes, PQ_nodes, V_batch)
                delta_p, delta_q, _, _ = func_deltas(buses_batch, gens_batch, lines_batch, V_batch,
                                                     self.slack_bus, self.slack_bus_gens)
                loss = self.cost_func(delta_p, delta_q)
            gradients = tape.gradient(loss, self.tgn_model.trainable_weights)
            gradients, _ = tf.clip_by_global_norm(gradients, glob_norm)
            self.optimizer.apply_gradients(zip(gradients, self.tgn_model.trainable_weights))

            tf.summary.scalar('loss', loss, step=self.current_train_iter)
            # Periodically log metrics and save model
            if ((save_step is not None) & (i % save_step == 0)) or (i == starting_point + max_iter - 1):
                # get minibatch train loss
                loss_final_train = loss
                print("loss", loss)
                # log metrics
                logging.info('    Learning iteration {}'.format(i))
                logging.info('    Training loss (minibatch) : {}'.format(loss_final_train))
                # save model
                self.save()

        # save model at the end of training
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def test(self):

        buses_batch, gens_batch, lines_batch, V_batch = self.data_importer.get_batch()
        n_nodes = V_batch['mag'].shape[1]
        PV_nodes, PV_nodes_l, PQ_nodes, PQ_nodes_l, S_nodes, S_nodes_l = self.determine_type(n_nodes, gens_batch['bus'])
        adj_matrices = self.sparse_adj_mat(lines_batch['from'], lines_batch['to'], n_nodes,
                                           PV_nodes, PQ_nodes, S_nodes)
        for tgn_layer in range(self.tgn_layers):
            nn_input = self.get_tgn_input(buses_batch, gens_batch, lines_batch, V_batch,
                                                  PV_nodes_l, PQ_nodes_l, S_nodes_l)
            deltaPV, deltaPQ = self.tgn_model(adj_matrices, nn_input)
            V_batch['mag'], V_batch['ang'] = self.update_V(deltaPV, deltaPQ, PV_nodes, PQ_nodes, V_batch)
        delta_p, delta_q, _, _ = func_deltas(buses_batch, gens_batch, lines_batch, V_batch,
                                             self.slack_bus, self.slack_bus_gens)
        loss_test = self.cost_func(delta_p, delta_q)
        return loss_test, Vmag_pred, Vang_pred, delta_p, delta_q
        
