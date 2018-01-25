import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
from tensorflow.python.framework import dtypes, tensor_shape

class TrainingHelper(Helper):
    def __init__(self, inputs, targets, output_dim):
        super(TrainingHelper, self).__init__()
        self._batch_size = tf.shape(inputs)[0]
        self._output_dim = output_dim
        self._targets = targets

        num_step = tf.shape(targets)[1]

        # tile and use the total num_step
        # use the whole frame and not mask it, because the 0 frames will stil contribute to silencing the audio
        self._length = tf.tile([num_step], [self._batch_size])

    def batch_size(self):
        return self._batch_size

    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    def sample_ids_dtype(self):
         return dtypes.int32

    def initialize(self, name=None):
        """return initial finished, initial inputs
            tacotron initial_inputs are go_frames """
        return tf.tile([False], [self._batch_size]),   tf.tile([0.0], [self._batch_size, self._output_dim])

    def sample(self, time, outputs, state, name=None):
        """ sample the ids """
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """ returns finished, next input, and state"""
        with tf.name_scope("training_helper"):
            finished = time+1 >= self._length
            next_input = self._targets[:, time, :]
            return finished, next_input, state
