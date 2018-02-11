import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
from tensorflow.python.framework import dtypes, tensor_shape

class TestingHelper(Helper):
    def __init__(self, batch_size, output_dim, r):
        self._batch_size = batch_size
        self._output_dim = output_dim
        self._end_token = tf.tile([0.0], [output_dim * r])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        """ return initial finished and initial inputs"""
        return tf.tile([False], [self._batch_size]), tf.tile([0.0], [self._batch_size, self._output_dim])

    def sample(self, time, outputs, state, name=None):
        """ sample the ids """
        return tf.tile([0.0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """ return finished, next input and state"""
        finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
        next_input = outputs[:, -self._output_dim:]
        return (finished, next_input, state)


class TrainingHelper(Helper):
    def __init__(self, inputs, targets, output_dim, r):
        super(TrainingHelper, self).__init__()
        self._batch_size = tf.shape(inputs)[0]
        self._output_dim = output_dim
        self._targets = targets[:, r-1::r, :]

        num_step = tf.shape(self._targets)[1]

        # tile and use the total num_step
        # use the whole frame and not mask it, because the 0 frames will stil contribute to silencing the audio
        self._length = tf.tile([num_step], [self._batch_size])
        # self._length = num_step

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
         return dtypes.int32

    def initialize(self, name=None):
        """return initial finished, initial inputs
            tacotron initial_inputs are go_frames """
        return tf.tile([False], [self._batch_size]),   tf.tile([[0.0]], [self._batch_size, self._output_dim])

    def sample(self, time, outputs, state, name=None):
        """ sample the ids """
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """ returns finished, next input, and state"""
        with tf.name_scope("training_helper"):
            finished = time+1 >= self._length
            next_input = self._targets[:, time, :]
            return finished, next_input, state
