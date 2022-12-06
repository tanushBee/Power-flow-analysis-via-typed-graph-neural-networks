import numpy as np
import tensorflow as tf
from misc import scatter_nd_add_diff, v_check_control

def func_deltas(buses, gens, lines, V, slack_b, slack_b_g, baseMVA=100):
    # buses: Pd, Qd, Gs, Bs [n_batch, n_nodes]
    # gens: Pg, Pmax, Pmin, mBase, bus [n_batch, n_gens]
    # lines: from, to, r, x, b, ratio [n_batch, n_lines]
    # V: mag, ang [n_batch, n_nodes]

    batch_size = buses['Pd'].shape[0]
    n_gens = gens['Pg'].shape[1]
    n_lines = lines['from'].shape[1]

    y_ij = 1. / tf.sqrt(lines['r'] ** 2 + lines['x'] ** 2)  
    delta_ij = tf.math.atan2(lines['r'], lines['x'])

    n_samples = tf.Variable(batch_size, trainable=False)
    dummy = tf.zeros_like(buses['Pd'])  

    # Build indices
    linspace = tf.expand_dims(tf.range(0, n_samples, 1), -1)  
    one_tensor = tf.ones([1], tf.int32)
    n_lines_tensor = tf.reshape(n_lines, [1]) 
    n_gens_tensor = tf.reshape(n_gens, [1])  
    shape_lines_indices = tf.concat([one_tensor, n_lines_tensor], axis=0) 
    shape_gens_indices = tf.concat([one_tensor, n_gens_tensor], axis=0)  

    indices_from = tf.reshape(tf.tile(linspace, shape_lines_indices), [-1])
    indices_from = tf.stack([indices_from, tf.reshape(lines['from'], [-1])], 1)

    indices_to = tf.reshape(tf.tile(linspace, shape_lines_indices), [-1])
    indices_to = tf.stack([indices_to, tf.reshape(lines['to'], [-1])], 1) 

    indices_gens = tf.reshape(tf.tile(linspace, shape_gens_indices), [-1])
    indices_gens = tf.stack([indices_gens, tf.reshape(gens['bus'], [-1])], 1)

    v_from = tf.gather(V['mag'], lines['from'], batch_dims=1)
    v_to = tf.gather(V['mag'], lines['to'], batch_dims=1)

    theta_from = tf.gather(V['ang'], lines['from'], batch_dims=1)
    theta_to = tf.gather(V['ang'], lines['to'], batch_dims=1)

    # power equations
    p_from_to = v_from * v_to * y_ij/lines['ratio'] * tf.math.sin(theta_from - theta_to - delta_ij) + \
                v_from**2 / lines['ratio']**2 * y_ij * tf.math.sin(delta_ij)

    p_to_from = v_to * v_from * y_ij/lines['ratio'] * tf.math.sin(theta_to - theta_from - delta_ij) + \
                v_to**2 * y_ij * tf.math.sin(delta_ij)

    q_from_to = - v_from * v_to * y_ij/lines['ratio'] * tf.math.cos(theta_from - theta_to - delta_ij) +\
                v_from**2/lines['ratio']**2 * (y_ij * tf.math.cos(delta_ij) - lines['b'] / 2)

    q_to_from = - v_to * v_from * y_ij/lines['ratio'] * tf.math.cos(theta_to - theta_from - delta_ij) +\
                v_to**2 * (y_ij * tf.math.cos(delta_ij) - lines['b'] / 2)

    # Active imbalance at each node 
    delta_p = - scatter_nd_add_diff(dummy, indices_from, tf.reshape(p_from_to, [-1])) \
              - scatter_nd_add_diff(dummy, indices_to, tf.reshape(p_to_from, [-1])) \
              - buses['Pd'] / baseMVA \
              - buses['Gs'] * V['mag']**2 / baseMVA

    # Reactive imbalance at each node 
    delta_q = - scatter_nd_add_diff(dummy, indices_from, tf.reshape(q_from_to, [-1])) \
              - scatter_nd_add_diff(dummy, indices_to, tf.reshape(q_to_from, [-1])) \
              - buses['Qd'] / baseMVA \
              + buses['Bs'] * V['mag']**2 / baseMVA

    # Qg locally compensated
    q_gens = tf.gather(-delta_q, gens['bus'], batch_dims=1)
    # Pg will be calculated for slack, nominal for other generators.
    slack_in_all = tf.zeros([batch_size, 1], dtype=tf.int32) + tf.constant([[slack_b]])
    slack_in_gens = tf.zeros([batch_size, 1], dtype=tf.int32) + tf.constant([[slack_b_g]])
    new_gens = tf.expand_dims(tf.transpose(gens['Pg']/baseMVA), 2)  
    pg_slack = tf.gather(-delta_p, slack_in_all, batch_dims=1) 
    dum = tf.gather(-gens['Pg']/baseMVA, slack_in_gens, batch_dims=1)  
    pg_slack = tf.expand_dims(pg_slack+dum, 0)  
    p_gens = tf.tensor_scatter_nd_add(new_gens, slack_gens, pg_slack)
    p_gens = tf.transpose(tf.squeeze(p_gens))

    delta_p += scatter_nd_add_diff(dummy, indices_gens, tf.reshape(p_gens, [-1]))
    delta_q += scatter_nd_add_diff(dummy, indices_gens, tf.reshape(q_gens, [-1]))
    p_gens = scatter_nd_add_diff(dummy, indices_gens, tf.reshape(p_gens, [-1]))
    q_gens = scatter_nd_add_diff(dummy, indices_gens, tf.reshape(q_gens, [-1]))

    return delta_p, delta_q, p_gens, q_gens
