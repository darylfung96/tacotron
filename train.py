import tensorflow as tf

from tacotron import Tacotron
from Hyperparameters import hp

"""
train data

"""



def main():
    inputs = tf.placeholder(tf.int32, shape=[None, None])
    mel_target = tf.placeholder(tf.float32, shape=[None, None, hp.num_mels])
    linear_target = tf.placeholder(tf.float32, shape=[None, None, hp.num_freq])

    Tacotron(inputs, mel_target, linear_target, batch_size=32)


if __name__ == '__main__':
    main()