import tensorflow as tf
import os
import pickle

from tacotron import Tacotron
from Hyperparameters import hp
from preprocess.preprocess_input import vectorize_input, _pad_input

"""
train data

"""


def get_data():
    inputs = []
    linear_targets = []
    mel_targets = []
    max_length = 0

    with open("training/train.txt", 'r') as f:
        for line in f:
            linear_target_filename, mel_target_filename, input = line.split('|')
            input = input.rstrip()
            input = vectorize_input(input)

            linear_target = load(linear_target_filename)
            mel_target = load(mel_target_filename)

            inputs.append(input); linear_targets.append(linear_target); mel_targets.append(mel_target)
            if len(input) > max_length:
                max_length = len(input)

    inputs = [_pad_input(input, max_length) for input in inputs]
    #TODO pad target to make them fix length when feeding into the model

    return inputs, linear_targets, mel_targets

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    tacotron = Tacotron(batch_size=32)
    inputs, linear_targets, mel_targets = get_data()
    tacotron.train(inputs, linear_targets, mel_targets)


if __name__ == '__main__':
    main()
