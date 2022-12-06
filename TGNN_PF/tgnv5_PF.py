import tensorflow as tf
from tensorflow.keras import layers

class FullyConnected(layers.Layer):
    def __init__(self,
                 latent_dimension=10,
                 hidden_layers=3,
                 non_lin='tanh',
                 output_dim=None,
                 name='fully_connected'):
        super(FullyConnected, self).__init__(name=name)
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        if non_lin == 'tanh':
            self.non_lin = tf.tanh
        elif non_lin == 'leaky_relu':
            self.non_lin = tf.nn.leaky_relu
        else:
            print("there is no activation function")
        self.output_dim = output_dim
        self._name = name

        # Build weights
        self.define_weights()

    def define_weights(self):
        """
        Builds the weights of the layer
        """

        # Initialize weights dict
        self.dense_ = {}

        # Iterate over all layers
        for layer in range(self.hidden_layers):

            if (layer == self.hidden_layers-1) and (self.output_dim is not None):
                self.dense_[str(layer)] = layers.Dense(units=self.output_dim,
                                                       name='dense_' + self._name + '_{}'.format(layer),
                                                       )
            else:
                self.dense_[str(layer)] = layers.Dense(units=self.latent_dimension,
                                                       activation=self.non_lin,
                                                       name='dense_' + self._name + '_{}'.format(layer),
                                                       )

    def __call__(self, h):
        for layer in range(self.hidden_layers):
            h = self.dense_[str(layer)](h)
        return h


class TGNmodel(tf.keras.Model):

    def __init__(self,
                 var,
                 mat,
                 msg,
                 loop,
                 in_dim,
                 out_dim,
                 time_steps,
                 non_lin='tanh',
                 name='tgn_model'):
        super(TGNmodel, self).__init__(name=name)

        self._name = name
        self.var, self.mat, self.msg, self.loop = var, mat, msg, loop
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.update_layers = 2
        self.msg_layers = 2
        self.time_steps = time_steps
        self.non_lin = non_lin

        # Check model for inconsistencies
        self.check_model()
        self.init_parameters()

    def init_parameters(self):

        self._encode = {
            v: FullyConnected(latent_dimension=d,
                              hidden_layers=1,
                              non_lin=self.non_lin,  # should i have act. function here or not??
                              output_dim=d,
                              name=v + '_encode_') for (v, d) in self.var.items()
        }

        self._update_MLPs = {
            v: FullyConnected(latent_dimension=d,
                              hidden_layers=self.update_layers,
                              non_lin=self.non_lin,
                              output_dim=d,
                              name=v + '_update_') for (v, d) in self.var.items()
        }

        self._msg_MLPs = {
            msg: FullyConnected(latent_dimension=self.var[vin],
                                hidden_layers=self.msg_layers,
                                non_lin=self.non_lin,  # should i have act. function here or not??
                                output_dim=self.var[vout],
                                name=msg + '_message_') for msg, (vin, vout) in self.msg.items()
        }

        self._decode = {
            v: FullyConnected(latent_dimension=d,
                              hidden_layers=1,
                              output_dim=self.output_dim[v],
                              name=v + '_decode_') for (v, d) in self.var.items()
        }

    def __call__(self, adj_matrices, initial_states):
        assertions = self.check_run(adj_matrices, initial_states)
        with tf.control_dependencies(assertions):
            states = {}
            for v, init in initial_states.items():
                states[v] = self._encode[v](h=init)
                states[v] = tf.cast(states[v], tf.float32)

            def while_body(t, states):
                new_states = {}
                for v in self.var:
                    inputs = []
                    for update in self.loop[v]:
                        if 'var' in update:
                            y = states[update['var']]
                            if 'fun' in update:
                                y = update['fun'](y)
                            # end if fun
                            if 'msg' in update:
                                y = self._msg_MLPs[update['msg']](h=y)
                            # end if msg
                            if 'mat' in update:
                                y = tf.sparse.sparse_dense_matmul(
                                    adj_matrices[update['mat']],
                                    y,
                                    adjoint_a=update['transpose?'] if 'transpose?' in update else False
                                )
                            # end if mat
                            inputs.append(y)
                        else:
                            inputs.append(adj_matrices[update['mat']])
                        # end if 'var' in update
                    # end for update in loop
                    inputs = tf.concat(inputs, axis=1)
                    if v != "V_S":
                        new_states[v] = self._update_MLPs[v](h=inputs)
                    else:
                        new_states[v] = states[v]
                    # end cell scope
                # end for v in var
                return (t+1), new_states
            # end while_body

            _, last_states = tf.while_loop(
                lambda t, states: tf.less(t, self.time_steps),
                while_body,
                [0, states]
            )

            output = {}
            for v, init in initial_states.items():
                if v != "V_E":
                    if v != "V_S":
                        output[v] = self._decode[v](h=last_states[v])

        return output['V_PV'], output['V_PQ']

    def check_run(self, adjacency_matrices, initial_states):
        assertions = []
        # Procedure to check model for inconsistencies
        num_vars = {}

        for v, d in self.var.items():
            init_shape = tf.shape(initial_states[v])  
            num_vars[v] = init_shape[0]

        for mat, (v1, v2) in self.mat.items():
            mat_shape = tf.shape(adjacency_matrices[mat])
            assertions.append(
                tf.assert_equal(
                    mat_shape[0],
                    num_vars[v1],
                    message="Matrix {m} doesn't have the same number of nodes as the initial embeddings of its variable {v}".format(
                        v=v1, m=mat)
                )
            )
            if type(v2) is int:
                assertions.append(
                    tf.assert_equal(
                        mat_shape[1],
                        v2,
                        message="Matrix {m} doesn't have the same dimensionality {d} on the second variable as declared".format(
                            m=mat,
                            d=v2
                        )
                    )
                )
            else:
                assertions.append(
                    tf.assert_equal(
                        mat_shape[1],
                        num_vars[v2],
                        message="Matrix {m} doesn't have the same number of nodes as the initial embeddings of its variable {v}".format(
                            v=v2,
                            m=mat
                        )
                    )
                )
            # end if-else
        # end for mat, (v1,v2)
        return assertions
    # end check_run

    def check_model(self):
        # Procedure to check model for inconsistencies
        for v in self.var:
            if v not in self.loop:
                raise Warning('Variable {v} is not updated anywhere! Consider removing it from the model'.format(v=v))
            # end if
        # end for

        for v in self.loop:
            if v not in self.var:
                raise Exception('Updating variable {v}, which has not been declared!'.format(v=v))
            # end if
        # end for

        for mat, (v1, v2) in self.mat.items():
            if v1 not in self.var:
                raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v1))
            # end if
            if v2 not in self.var and type(v2) is not int:
                raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v2))
            # end if
        # end for

        for msg, (v1, v2) in self.msg.items():
            if v1 not in self.var:
                raise Exception('Message {msg} maps from undeclared variable {v}'.format(msg=msg, v=v1))
            # end if
            if v2 not in self.var:
                raise Exception('Message {msg} maps to undeclared variable {v}'.format(msg=msg, v=v2))
            # end if
        # end for
    # end check_model

    # end TGN
