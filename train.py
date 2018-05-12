import pickle
import linecache
from tacotron import Tacotron
from preprocess.preprocess_input import vectorize_input, pad_input, pad_target

"""
train data

"""


def get_data(starting_index):
    inputs = []
    linear_targets = []
    mel_targets = []
    max_input_length = 0
    max_target_length = 0
    index = 0
    with open("training/train.txt", 'r') as f:
        for line in f.readlines()[starting_index:starting_index+1]:
            index += 1
            linear_target_filename, mel_target_filename, input = line.split('|')
            input = vectorize_input(input)

            linear_target = load(linear_target_filename)
            mel_target = load(mel_target_filename)

            inputs.append(input); linear_targets.append(linear_target); mel_targets.append(mel_target)

            if len(input) > max_input_length: max_input_length = len(input)
            if len(mel_target) > max_target_length: max_target_length = len(mel_target)
            if index > 2500: break

    inputs = [pad_input(input, max_input_length) for input in inputs]
    linear_targets = [pad_target(linear_target, max_target_length) for linear_target in linear_targets]
    mel_targets = [pad_target(mel_target, max_target_length) for mel_target in mel_targets]

    return inputs, linear_targets, mel_targets


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    tacotron = Tacotron(batch_size=64)
    num_lines = sum(1 for _ in open("training/train.txt"))
    max_index = num_lines//64

    while True:
        index = 0
        inputs, linear_targets, mel_targets = get_data(index)
        tacotron.train(inputs, linear_targets, mel_targets)
        index += 1
        if index >= max_index:
            index = 0


if __name__ == '__main__':
    main()
