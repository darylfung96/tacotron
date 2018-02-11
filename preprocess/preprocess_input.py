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
    # max_length+1 because max_length will be hard to divide into r_frames and converted to match
    # the target output later on.
    # eg. 799/3 = 266
    # 266 * 3 = 798     -> does not match 799. So we add max_length + 1 to simplify this.
    target = np.pad(target, [(0, max_length+1 - len(target)), (0, 0)], mode='constant', constant_values=0)
    return target



# convert number to words
import num2words

