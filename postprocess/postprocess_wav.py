import numpy as np

from Hyperparameters import hp


def inv_spectrogram(spectrogram):
    # value =  db to amp (denormalize + ref level db)
    #inv_preemphasis (griffin(value ** harmonic)) (harmonic is to increase the beauty if the sound)
    pass


def _db_to_amp(inputs):
    return np.power(10, inputs/20)

def _denormalize(inputs):
    return (np.clip(inputs, 0, 1) * -hp.min_level_db) + hp.min_level_db