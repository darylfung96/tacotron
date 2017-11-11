import tensorflow as tf
import numpy as np

# declare weights and biases
def declare_layer(inputs, num_outputs, name):
    input_shape = inputs.get_shape().as_list()[-1]

    bias_shape = [num_outputs]

    value_bound = np.sqrt(6./(input_shape+num_outputs))
    initial_weight_value = tf.random_uniform(shape=[input_shape, num_outputs], minval=-value_bound, maxval=value_bound)
    initial_bias_value = tf.zeros(shape=bias_shape)

    weights = tf.get_variable(name+"_weights", initializer=initial_weight_value)
    biases = tf.get_variable(name+"_bias", initializer=initial_bias_value)


    outputs = tf.nn.relu(tf.matmul(inputs, weights) + biases)


    return outputs
