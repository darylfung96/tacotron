"""
    prenet
    |
    V
    attention
    |
    V
    decoder GRU ( 2 layers )
    |
    V
    split to r values
    |
    V
    griffin
    
    
    we should have two decoders, for the loss function since there are two loss functions


"""
from tensorflow.contrib.rnn import GRUCell

from network_module import prenet

def prenet(inputs):
    with tf.variable_scope("prenet"):
        layer1 = tf.nn.dropout(tf.layers.dense(inputs, 256, activation=tf.nn.relu), keep_prob=0.5)
        layer2 = tf.nn.dropout(tf.layers.dense(layer1, 128, activation=tf.nn.relu), keep_prob=0.5)
    return layer2


def attention_decoder(inputs, memory, num_units):

    decoder_cell = tf.contrib.rnn.GRUCell(num_units)

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory, normalize=True,
        probability_fn=tf.nn.softmax)  # probability_fn uses Softmax function to get the weights to focus on the memory
    dec_cells_and_attentions = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, num_units)

    outputs, _ = tf.nn.dynamic_cell(dec_cells_and_attentions, inputs, dtype=tf.float32)

    return outputs


def first_decoding(inputs, encoder_inputs):
    outputs = prenet(inputs)

    outputs = attention_decoder(inputs, encoder_inputs, 256)

    gru1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(128), GRUCell(128), outputs, dtype=tf.float32)
    gru2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(128), GRUCell(128), outputs, dtype=tf.float32)

    outputs += gru1
    outputs += gru2

    out_dim = inputs.get_shape().as_list()[-1]

    return tf.layers.dense(outputs, out_dim)