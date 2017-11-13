"""
Encoder:
    text
    |
    V
    embedding
    |
    V
    Prenet
    |
    V
    CBHG
    |
    V
    residual connection
    |
    V
    GRU bidirectional
    

"""
import tensorflow as tf
import numpy as np
import torch
from initializer_util import declare_layer


def embed(inputs, vocab_size, num_units=256, zero_pad=True):
    with tf.variable_scope("embedding"):
        embedding_table = tf.get_variable("embedding_table", shape=[vocab_size, num_units], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        if zero_pad:
            embedding_table = tf.concat((tf.zeros(shape=[1, num_units]), embedding_table[1:, :]), 0)

    return tf.nn.embedding_lookup(embedding_table, inputs)


def prenet(inputs):
    with tf.variable_scope("prenet"):
        layer1 = tf.nn.dropout(tf.layers.dense(inputs, 256, activation=tf.nn.relu), keep_prob=0.5)
        layer2 = tf.nn.dropout(tf.layers.dense(layer1, 128, activation=tf.nn.relu), keep_prob=0.5)
    return layer2


"""
    Convolution bank + pooling + Highway network + GRU (CBHG)
"""

# conv1d bank
def conv1dbank(inputs, k):
    outputs = conv1d(inputs, filters=128, kernel_size=1, activation=tf.nn.relu)
    for i in range(2, k+1):
        single_output = conv1d(outputs, 128, k, activation=tf.nn.relu)
        outputs = tf.concat((outputs, single_output), -1)
    return tf.layers.batch_normalization(outputs, training=True, epsilon=1e-7)


""" 
highway network 

T = transform gate
C = carry the original value

C = 1. - T

Notice that when T = 0, output is input.
So when we completely transform it, we don't let any input pass through it.

output = H * T + input * C

"""
def highwaynet(inputs, num_units):

    H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="H_encoder_highway")
    T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, name="T_encoder_highway")

    C = 1. - T

    outputs = H * T + inputs * C
    return outputs



""" end of CBHG helper methods """


#CBHG
def cbhg(inputs, k):
    outputs = embed(inputs, 0)          # N, text_size, em_size
    prenet_outputs = prenet(outputs)
    outputs = conv1dbank(prenet_outputs, k)    # N, text_size, k * em_size/2
    #pooling
    outputs = tf.layers.max_pooling1d(outputs, 2, 1, padding='same')    # same size (N, text_size, k * em_size/2)
    #conv1d projection
    outputs = tf.layers.conv1d(outputs, 128, 3)                         # N, text_size, 128
    outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=True))
    outputs = tf.layers.conv1d(outputs, 128, 3)                         # N, text_size, 128
    outputs = tf.layers.batch_normalization(outputs, training=True)
    #add residual connection
    outputs += prenet_outputs

    #TODO:  GRU
    outputs = highwaynet(outputs, 128)
    pass



def encoder(inputs):
    pass





def conv1d(inputs, filters, kernel_size, activation):
    outputs = tf.layers.conv1d(inputs, filters=filters, kernel_size=kernel_size, activation=activation)
    return tf.layers.batch_normalization(outputs, training=True)

#  toy data
data = np.array(np.random.rand(100, 26), dtype=np.float32)
data = tf.convert_to_tensor(data, dtype=tf.float32)
prenet(data)