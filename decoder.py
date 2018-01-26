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
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, RNNCell, OutputProjectionWrapper, MultiRNNCell, ResidualWrapper
from tensorflow.contrib.seq2seq import Helper, dynamic_decode, BasicDecoder

from network_module import prenet, cbhg
from Hyperparameters import hp

from RNNHelper import TrainingHelper


def attention_decoder(inputs, memory, num_units):

    decoder_cell = tf.contrib.rnn.GRUCell(num_units)

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, memory, normalize=True,
        probability_fn=tf.nn.softmax)  # probability_fn uses Softmax function to get the weights to focus on the memory
    dec_cells_and_attentions = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, num_units)

    outputs, _ = tf.nn.dynamic_rnn(dec_cells_and_attentions, inputs, dtype=tf.float32)

    return outputs


def first_decoding(inputs, encoder_inputs):
    outputs = prenet(inputs)

    outputs = attention_decoder(outputs, encoder_inputs, 256)

    gru1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(128), GRUCell(128), outputs, dtype=tf.float32)
    gru2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(128), GRUCell(128), outputs, dtype=tf.float32)

    outputs += gru1
    outputs += gru2

    out_dim = inputs.get_shape().as_list()[-1]
    outputs = tf.layers.dense(outputs, out_dim * 3)  # 3 is the reduction factor
    return outputs

def second_decoding(inputs):
    outputs = prenet(inputs)

    outputs = cbhg(outputs, 16)

    out_dim = 1+2048//2        # 2048 = sample points, 3 = r (reduction)

    return tf.layers.dense(outputs, out_dim)


""" decoding wrappers """
class DecoderPrenetWrapper(RNNCell):

    def __init__(self, cell, is_training):
        super(DecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training

    def state_size(self):
        return self._cell.state_size

    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        prenet_outputs = prenet(inputs, self._is_training)
        return self._cell(prenet_outputs, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatAttentionOutputWrapper(RNNCell):
    def __init__(self, cell):
        super(ConcatAttentionOutputWrapper, self).__init__()
        self._cell = cell

    def state_size(self):
        return self._cell.state_size

    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def __call__(self, inputs, state, scope=None):
        output, state = self._cell(inputs, state)
        return tf.concat([output, state.attention], axis=-1), state





def full_decoding(encoder_outputs, is_training, inputs, mel_targets, batch_size=32):
    # prenet
    # attention
    # concat attention
    # gru gru
    # output

    # inside cell_outputs, there's the states that contain the attentions
    # We make DecoderPrenetWrapper and ConcatAttentionWrapper an RNNCell so that
    # later on there is a helper function that can do everything for us right away
    cell_outputs = tf.contrib.seq2seq.AttentionWrapper(
        DecoderPrenetWrapper(GRUCell(256), is_training=is_training),
        tf.contrib.seq2seq.BahdanauAttention(256, encoder_outputs, normalize=True, probability_fn=tf.nn.softmax),
        alignment_history=True,
        output_attention=False
    )
    """
        I learn that in AttentionWrapper, if we set alignment_history to true, we will get all the previous alignments
        If we set the output_attention to false, we will get an output for the cell_output instead of the attention, but 
        the attention will be stored in the state.
        
        We do this so we can combine the output the attention.
    """

    output_attention_cell = ConcatAttentionOutputWrapper(cell_outputs)

    decoding_cells = MultiRNNCell([
        OutputProjectionWrapper(output_attention_cell, 256),
        ResidualWrapper(GRUCell(256)),
        ResidualWrapper(GRUCell(256))
    ])

    decoder_outputs = OutputProjectionWrapper(decoding_cells, hp.num_mels*hp.r_frames)
    decoder_initial_states = decoder_outputs.zero_state(batch_size=batch_size, dtype=tf.float32)


    #TODO: testing helper function
    if is_training:
        helper = TrainingHelper(inputs, mel_targets, hp.num_mels, hp.r_frames)

    #TODO specify the return values for dynamic_decode
    dynamic_decode(BasicDecoder(decoder_outputs, helper, decoder_initial_states))


    pass