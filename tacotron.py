import tensorflow as tf

from encoder import encoder
from decoder import full_decoding
from Hyperparameters import hp
from postprocess.postprocess_wav import inv_spectrogram, save_audio



class Tacotron:
    def __init__(self, batch_size, is_training=True):
        self.inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.mel_targets = tf.placeholder(tf.float32, shape=[None, None, hp.num_mels])
        self.linear_targets = tf.placeholder(tf.float32, shape=[None, None, hp.num_freq/2 + 1])

        self._batch_size = batch_size
        self._is_training = is_training

        self.embedding_variables = tf.get_variable('embedding', shape=[len(hp.symbols), 256])
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_variables, self.inputs)

        self.encoder_outputs = encoder(self.embedding_inputs, is_training=is_training)
        self.mel_outputs, self.linear_outputs = full_decoding(self.inputs, self.encoder_outputs, is_training, self.mel_targets, batch_size=batch_size)

        self._loss()
        self._optimizer()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.current_step = 0


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

    def train(self, inputs, linear_targets, mel_targets):
        loss, linear_output, _ = self.sess.run([self.total_loss, self.linear_outputs, self.train_op], feed_dict={
            self.inputs: inputs,
            self.mel_targets: mel_targets,
            self.linear_targets: linear_targets
        })

        waveform = inv_spectrogram(linear_output.T)
        save_audio(waveform, 'audio/test.wav')

        self.current_step += 1

        print("iteration {} loss: {}".format(self.current_step, loss))

        if self.current_step % 50 == 0:
            print('loss at {} step: {}'.format(self.current_step, loss))
