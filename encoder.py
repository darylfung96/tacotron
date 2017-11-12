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


#CBHG
def cbhg(inputs, k):
    outputs = conv1dbank(inputs, k)

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