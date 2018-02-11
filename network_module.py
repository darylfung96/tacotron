import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

#TODO: change training value in batch normalization

def embed(inputs, vocab_size, num_units=256, zero_pad=True):
    with tf.variable_scope("embedding"):
        embedding_table = tf.get_variable("embedding_table", shape=[vocab_size, num_units], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        if zero_pad:
            embedding_table = tf.concat((tf.zeros(shape=[1, num_units]), embedding_table[1:, :]), 0)

    return tf.nn.embedding_lookup(embedding_table, inputs)


def prenet(inputs, is_training, scope=None):
    dropout_rate = 0.5 if is_training else 0
    with tf.variable_scope(scope or "prenet"):
        layer1 = tf.nn.dropout(tf.layers.dense(inputs, 256, activation=tf.nn.relu), keep_prob=dropout_rate)
        layer2 = tf.nn.dropout(tf.layers.dense(layer1, 128, activation=tf.nn.relu), keep_prob=dropout_rate)
    return layer2


# convolution 1 dimension that includes the batch normalization
def conv1d(inputs, filters, kernel_size, activation, is_training=True):
    outputs = tf.layers.conv1d(inputs, filters=filters, kernel_size=kernel_size, activation=activation, padding='same')
    return tf.layers.batch_normalization(outputs, training=is_training)

"""
    Convolution bank + pooling + Highway network + GRU (CBHG)
"""

# conv1d bank
def conv1dbank(inputs, k, is_training=True):
    """
    We want to get the features from the input here (128 units).
    We passed in a filter of conv1d to extract the feature of this input and
    concat the filters together.
    Each filter will expand to the whole 128 units and extract feature. The last output will be 1 if there is only 1 filter,
    2 if there are 2 filters...

    :param inputs:
    :param k:
    :return:
    """
    outputs = conv1d(inputs, filters=1, kernel_size=128, activation=tf.nn.relu, is_training=is_training)
    for i in range(2, k+1):
        output = conv1d(inputs, filters=i, kernel_size=128, activation=tf.nn.relu, is_training=is_training)
        outputs = tf.concat((outputs, output), -1)
    return tf.layers.batch_normalization(outputs, training=is_training, epsilon=1e-7)


""" 
highway network 

T = transform gate
C = carry the original value

C = 1. - T

Notice that when T = 0, output is input.
So when we completely transform it, we don't let any input pass through it.

output = H * T + input * C

"""
def highwaynet(inputs, num_units, scope=None):

    H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="H_encoder_highway_"+scope)
    T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, name="T_encoder_highway_"+scope)

    C = 1. - T

    outputs = H * T + inputs * C
    return outputs



""" end of CBHG helper methods """


#CBHG
"""
CBHG:
        conv1d bank with 128 neurons and K(1 to K) times features mapping
        pooling layer with stride 1 and width 2
        conv1d projections with 128 neurons (2 of them)
        residual connection from conv1bank
        highway network (4 layers)
        GRU bi-directinal
"""
def cbhg(inputs, k, projections=[128, 128], scope=None, is_training=True):

    with tf.variable_scope(scope):
        outputs = conv1dbank(inputs, k, is_training=is_training)    # N, text_size, k * embedding_size/2
        #pooling
        outputs = tf.layers.max_pooling1d(outputs, 2, 1, padding='same')    # same size (N, text_size, k * embedding_size/2)
        #conv1d projection
        outputs = conv1d(outputs, filters=projections[0], kernel_size=3, activation=tf.nn.relu, is_training=is_training)    # N, text_size, 128(embedding_size/2)
        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_training))
        outputs = conv1d(outputs, filters=projections[1], kernel_size=3, activation=tf.nn.relu, is_training=is_training)    # N, text_size, 128(embedding_size/2)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
        #add residual connection
        outputs += inputs       # error here: 157,37,128 + 157,168,128

        # fix wrong dimension
        if outputs[2] != 128:
            outputs = tf.layers.dense(outputs, 128)

        #highway networks
        for i in range(4):  # 4 highway networks just like in the paper
            outputs = highwaynet(outputs, 128, '{}_{}'.format(scope, i))

        #bi-directional GRU
        outputs, states = tf.nn.bidirectional_dynamic_rnn(GRUCell(128), GRUCell(128), outputs, dtype=tf.float32)
        outputs = tf.concat(outputs, 2)  # combine the forward and backward GRU #N, text_size, embedding_size

        return outputs