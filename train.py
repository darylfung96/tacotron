import tensorflow as tf

from tacotron import Tacotron

"""
train data

"""



def main():
    inputs = tf.placeholder(tf.float32, shape=[None, None])
    mel_target = tf.placeholder(tf.float32, shape=[None, None])
    linear_target = tf.placeholder(tf.float32, shape=[None, None])

    Tacotron(inputs, mel_target, linear_target, batch_size=inputs[0])


if __name__ == '__main__':
    main()