import numpy as np

from Hyperparameters import hp


def _vectorize_inputs(inputs):
    return [vectorize_input(input) for input in inputs]

def vectorize_input(input):
    return [hp.symbols.index(character) for character in input]

def _pad_input(x, max_length):
    x = np.pad(x, (0, max_length - len(x)), mode='constant', constant_values=0)
    return x


def _pad_target(target, max_length):
    pass