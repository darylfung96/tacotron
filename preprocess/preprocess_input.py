import numpy as np

from Hyperparameters import hp

"""
np.pad(input, [(up, down),(left, right)], mode, value)
"""

def _vectorize_inputs(inputs):
    return [vectorize_input(input) for input in inputs]

def vectorize_input(input):
    input = input.rstrip()
    return [hp.symbols.index(character) for character in input]

def pad_input(x, max_length):
    x = np.pad(x, (0, max_length - len(x)), mode='constant', constant_values=0)
    return x

def pad_target(target, max_length):
    target = np.pad(target, [(0, max_length - len(target)), (0, 0)], mode='constant', constant_values=0)
    return target