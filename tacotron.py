import tensorflow as tf

from encoder import encoder
from decoder import full_decoding
from Hyperparameters import hp

symbols = '_' + '~' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz' + '!\'(),-.:;? '


class Tacotron:
    def __init__(self, inputs, mel_targets, linear_targets, batch_size, is_training=True):
        self._batch_size = batch_size
        self._is_training = is_training
        self.mel_targets = mel_targets
        self.linear_targets = linear_targets

        self.embedding_variables = tf.get_variable('embedding', shape=[len(symbols), 256])
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_variables, inputs)

        self.encoder_outputs = encoder(self.embedding_inputs)
        self.mel_outputs, self.linear_outputs = full_decoding(self.encoder_outputs, is_training, mel_targets, batch_size=batch_size)


    def _loss(self):
        self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
        linear_loss = tf.abs(self.linear_targets - self.linear_outputs)

        # we want to do prioritize training, so focus on frequency with 3000HZ or lower
        priority_loss = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
        self.linear_loss = 0.5 * tf.reduce_mean(linear_loss) + 0.5 * tf.reduce_mean(linear_loss[:, :, :priority_loss])

        self.total_loss = self.mel_loss + self.linear_loss

    #TODO might need momentum
    def _optimizer(self):
        self.optimizer = tf.train.AdamOptimizer()
        self.tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, self.tvars), 1.0)
        self.train_op = self.optimizer.apply_gradients(zip(grads, self.tvars))