import numpy as np


def _pad_input(x, max_length):
    x = np.pad(x, (0, max_length - len(x)), mode='constant', constant_values=0)
    return x


def _pad_target(target, max_length):
    pass