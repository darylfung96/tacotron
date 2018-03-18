import librosa
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


def _griffin_lim(spectrogram, num_iterations):
    """
    Given spectrogram, recover audio from magnitude only spectrogram.
    - Minimizes mean squared error between STFT of estimated signal and the modified STFT.
    - Use iterative algorithm to estimate a signal from its modified STFT magnitude.
    - Mean squared error between STFT magnitude of estimated signal and the modified STFT magnitude is reduced in
      each iteration.


    :param spectrogram:
    :return: audio
    """
    angles = np.exp(2 * np.pi * np.random.rand(*spectrogram.shape) )
    hop_length = hp.frame_shift / 1000 * hp.sample_rate
    window_length = hp.frame_length / 1000 * hp.sample_rate
    for i in range(num_iterations):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, win_length=window_length, window='hann')
        rebuild = librosa.stft(inverse, hop_length=hop_length, win_length=window_length, window='hann')
        angles = np.exp(1j * np.angle(rebuild))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, win_length=window_length, window='hann')
    return inverse
