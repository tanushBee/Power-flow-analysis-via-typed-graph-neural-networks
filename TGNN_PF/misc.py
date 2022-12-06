import tensorflow as tf


@tf.custom_gradient
def scatter_nd_add_diff(x0, x1, x2):
    """
    Custom gradient version of scatter_nd_add
    """
    dummy = tf.Variable(x0, name='dummy') #use_resource should not be needed in tensorflow 2.0
    reset_dummy = dummy.assign(0.0*x0)
    #reset_dummy = 0.0*x0

    with tf.control_dependencies([reset_dummy]): #wrapper for Graph.control_dependencies() using the default graph. might've to change: https://www.tensorflow.org/api_docs/python/tf/Graph#control_dependencies
        #add updates x2, to tensor dummy at indices x1.
        f = tf.tensor_scatter_nd_add(dummy, x1, x2)

    #define how the gradient is to be calculated, dy = gradient (normal).
    def grad(dy, variables=[dummy]):
        g = tf.gather_nd(dy, x1) #dy values corresponding to the ones specified by indices
        return [None, None, g] , [None] #why NONES with this form???????????????
    return f, grad

@tf.custom_gradient
def v_check_control(V_update, V_setpoint, gen_indices):
    sparse_gen = tf.sparse.SparseTensor(indices=gen_indices, values=[0.0] * gen_indices.shape[0],
                                        dense_shape=V_update.get_shape())
    gen_mask = tf.sparse.to_dense(sparse_gen, default_value=1.0, validate_indices=False)
    sparse_value = tf.sparse.SparseTensor(indices=gen_indices, values=tf.reshape(V_setpoint, [-1]),
                                          dense_shape=V_update.get_shape())
    value_mask = tf.sparse.to_dense(sparse_value, default_value=0.0, validate_indices=False)
    f = gen_mask * V_update + value_mask #[n_samples, 2*n_buses]

    def grad(dy):
        # sparse_g1 = tf.sparse.SparseTensor(indices=gen_indices, values=[0.0]*len(gen_indices), dense_shape=V_update.get_shape())
        sparse_g1 = tf.sparse.SparseTensor(indices=gen_indices, values=[0.0] * gen_indices.shape[0],
                                           dense_shape=V_update.get_shape())
        g1 = tf.sparse.to_dense(sparse_g1, default_value=1.0, validate_indices=False)
        #dy of non-gen buses
        g1 = g1 * dy
        #dy values corresponding to gen buses
        g2 = tf.gather_nd(dy, gen_indices)
        g2 = tf.reshape(g2, tf.shape(V_setpoint))
        return [g1, g2, None] #why are 3 gradients expected????
    return f, grad

@tf.custom_gradient
def s_check_control(Th_update, s_setpoint=0, s_indices=0):
    sparse_gen = tf.sparse.SparseTensor(indices=s_indices, values=[0.0] * s_indices.shape[0],
                                        dense_shape=Th_update.get_shape())
    gen_mask = tf.sparse.to_dense(sparse_gen, default_value=1.0, validate_indices=False)
    sparse_value = tf.sparse.SparseTensor(indices=s_indices, values=tf.reshape(s_setpoint, [-1]),
                                          dense_shape=Th_update.get_shape())
    value_mask = tf.sparse.to_dense(sparse_value, default_value=0.0, validate_indices=False)
    f = gen_mask * Th_update + value_mask

    def grad(dy):
        sparse_g1 = tf.sparse.SparseTensor(indices=s_indices, values=[0.0] * s_indices.shape[0],
                                           dense_shape=Th_update.get_shape())
        g1 = tf.sparse.to_dense(sparse_g1, default_value=1.0, validate_indices=False)
        #dy of non-gen buses
        g1 = g1 * dy
        #dy values corresponding to gen buses
        g2 = tf.gather_nd(dy, s_indices)
        g2 = tf.reshape(g2, tf.shape(s_setpoint))
        return [g1, g2, None]
    return f, grad
