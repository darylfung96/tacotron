import tensorflow as tf
import os
import logging

from encoder import encoder
from decoder import full_decoding
from Hyperparameters import hp
from postprocess.postprocess_wav import inv_spectrogram, save_audio


class Tacotron:
    def __init__(self, batch_size, is_training=True, save_step=50):
        self.logger = logging.getLogger(__name__)
        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter('%(asctime)s - %(name)s: %(message)s')
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)

        self.inputs = tf.placeholder(tf.int32, shape=[None, None])
        self.mel_targets = tf.placeholder(tf.float32, shape=[None, None, hp.num_mels])
        self.linear_targets = tf.placeholder(tf.float32, shape=[None, None, hp.num_freq/2 + 1])

        self._batch_size = batch_size
        self._is_training = is_training
        self._save_step = save_step

        with tf.device('/gpu:0'):
            self.embedding_variables = tf.get_variable('embedding', shape=[len(hp.symbols), 256])
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_variables, self.inputs)

            self.encoder_outputs = encoder(self.embedding_inputs, is_training=is_training)
            self.mel_outputs, self.linear_outputs = full_decoding(self.inputs, self.encoder_outputs, is_training, self.mel_targets, batch_size=batch_size)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self._loss()
        self._optimizer()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        self._summary_graph()
        self._make_model_dir()
        self._load_model()

    def _loss(self):
        self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
        linear_loss = tf.abs(self.linear_targets - self.linear_outputs)

        # we want to do prioritize training, so focus on frequency with 3000HZ or lower
        priority_loss = int(3000 / (hp.sample_rate * 0.5) * (hp.num_freq/2+1) )
        self.linear_loss = 0.5 * tf.reduce_mean(linear_loss) + 0.5 * tf.reduce_mean(linear_loss[:, :, :priority_loss])

        self.total_loss = self.mel_loss + self.linear_loss

    def _make_model_dir(self):
        if not os.path.isdir(hp.model_dir):
            os.mkdir(hp.model_dir)

    def _load_model(self):
        self.logger.info('loading model...')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(hp.model_dir))
        logging.info('successfully loaded model')

    # TODO might need momentum
    def _optimizer(self):
        self.optimizer = tf.train.AdamOptimizer()
        self.tvars = tf.trainable_variables()

        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, self.tvars), 1.0)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars), global_step=self.global_step)

    def _summary_graph(self):
        self.loss_summary = tf.placeholder(tf.float32)

        for grad, tvar in zip(self.grads, self.tvars):
            tf.summary.histogram(tvar.name+'/grad', grad)
            tf.summary.histogram(tvar.name, tvar)

        tf.summary.scalar('training_loss_graph', self.loss_summary)
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter('logs', self.sess.graph)

    def train(self, inputs, linear_targets, mel_targets):
        feed_dict = {
            self.inputs: inputs,
            self.mel_targets: mel_targets,
            self.linear_targets: linear_targets
        }
        current_global_step = self.sess.run(self.global_step)

        loss, linear_output, _ = self.sess.run([self.total_loss, self.linear_outputs, self.train_op], feed_dict=feed_dict)

        waveform = inv_spectrogram(linear_output[0].T)

        if current_global_step % hp.save_audio_every_ter == 0:
            save_audio(waveform, 'audio/test.wav')
            save_audio(inv_spectrogram(linear_targets[0].T), 'audio/target.wav')

        self.logger.info("iteration {} loss: {}".format(current_global_step, loss))

        if self.sess.run(self.global_step) % 100 == 0:
            feed_dict.update({self.loss_summary: loss})
            summary = self.sess.run(self.merged, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, current_global_step)
            self.saver.save(self.sess, os.path.join(hp.model_dir, hp.model_file_name), current_global_step)
