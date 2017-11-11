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


def embed(inputs, vocab_size, num_units=256, zero_pad=True):
    with tf.variable_scope("embedding"):
        embedding_table = tf.get_variable("embedding_table", shape=[vocab_size, num_units], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        if zero_pad:
            embedding_table = tf.concat((tf.zeros(shape=[1, num_units]), embedding_table[1:, :]), 0)

    return tf.nn.embedding_lookup(embedding_table, vocab_size)


def prenet(inputs, num_units):
    with tf.variable_scope("prenet"):
        layer1 = tf.layers.dense(inputs, num_units, activation=tf.nn.relu, name="layer1")
        layer1 = tf.nn.dropout(layer1, keep_prob=0.5, name="layer1_with_dropout")
        layer2 = tf.layers.dense(layer1, num_units, activation=tf.nn.relu, name="layer2")
        layer2 = tf.nn.dropout(layer2, keep_prob=0.5, name="layer2_with_dropout")
    return layer2


def cbhg(inputs):
    pass



def encoder(inputs):
    pass