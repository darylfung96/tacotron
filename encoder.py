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
from initializer_util import declare_layer


def embed(inputs, vocab_size, num_units=256, zero_pad=True):
    with tf.variable_scope("embedding"):
        embedding_table = tf.get_variable("embedding_table", shape=[vocab_size, num_units], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        if zero_pad:
            embedding_table = tf.concat((tf.zeros(shape=[1, num_units]), embedding_table[1:, :]), 0)

    return tf.nn.embedding_lookup(embedding_table, vocab_size)


def prenet(inputs):
    with tf.variable_scope("prenet"):
        layer1 = declare_layer(inputs, 256, "layer1")
        layer2 = declare_layer(layer1, 128, "layer2")
    return layer2


def cbhg(inputs):
    pass



def encoder(inputs):
    pass



#  toy data
#TODO make it available for 3D
data = np.array(np.random.rand(100, 26), dtype=np.float32)
data = tf.convert_to_tensor(data, dtype=tf.float32)
prenet(data)