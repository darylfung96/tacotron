import numpy as np
import scipy.signal as signal

from Hyperparameters import hp


def inv_spectrogram(spectrogram):
    # value =  db to amp (denormalize + ref level db)
    #inv_preemphasis (griffin(value ** harmonic)) (harmonic is to increase the beauty if the sound)
    pass


def _db_to_amp(inputs):
    return np.power(10, inputs/20)

def _denormalize(inputs):
    return (np.clip(inputs, 0, 1) * -hp.min_level_db) + hp.min_level_db

def _inv_preemphasis(inputs):
    return signal.lfilter([1], [1, -hp.pre_emphasis], inputs)

#TODO griffin lim algorithm
def _griffin_lim(spectrogram):
    """
    Given spectrogram, recover audio from magnitude only spectrogram.
    - Minimizes mean squared error between STFT of estimated signal and the modified STFT.
    - Use iterative algorithm to estimate a signal from its modified STFT magnitude.
    - Mean squared error between STFT magnitude of estimated signal and the modified STFT magnitude is reduced in
      each iteration.


    :param spectrogram:
    :return: audio
    """
    pass