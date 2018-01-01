import librosa
from scipy import signal
import numpy as np

from Hyperparameters import hp

"""

To preprocess data:
1.  load wav file data 
2.  pre-emphasis it 
3.  convert it into stft (short-time Fourier Transform) with Hann windowing (becomes linear), might want to convert 
    to mel from linear if melspectrogram is what we want
4.  convert the amplitude to db
5.  normalize it



"""

def process_wav(wav_file):
    wav = librosa.core.load(wav_file, sr=hp.sample_rate)[0]  # load as floating point time series

    spec_wav = _get_spectrogram(wav)
    mel_wav = _get_mel_spectrogram(wav)

    return spec_wav, mel_wav


def _pre_emphasis(inputs):
    return signal.lfilter([1, -hp.pre_emphasis], [1], inputs)

def _stft(inputs):
    hop_length = int(hp.frame_shift/1000 * hp.sample_rate)      # frame shift(second) * sample rate(HZ/second)
    win_length = int(hp.frame_length/1000 * hp.sample_rate)     # frame length(second) * sample rate(HZ/second)
    return librosa.stft(inputs, n_fft=hp.num_freq, hop_length=hop_length, win_length=win_length, window='hann')

def _amp_to_db(inputs):
    return 20 * np.log10(np.maximum(1e-5, inputs))      # [1e-6, 1e-7, 10, 5] -> [1e-5, 1e-5, 10, 5]

def _linear_to_mel(spectrogram):
    mel_filter = librosa.filters.mel(hp.sample_rate, hp.num_freq, n_mels=hp.num_mels)
    return np.dot(mel_filter, spectrogram)

# normalize value based on min-level db percentage
def _normalize(inputs):
    return np.clip( (inputs - hp.min_level_db) / -hp.min_level_db, 0, 1)

def _denormalize(inputs):
    return np.clip(inputs, 0, 1) * -hp.min_level_db + hp.min_level_db


def _get_spectrogram(inputs):
    D = _stft(_pre_emphasis(inputs))
    D = np.abs(D)
    outputs = _amp_to_db(D) - hp.amp_reference
    return _normalize(outputs)

def _get_mel_spectrogram(inputs):
    D = _stft(_pre_emphasis(inputs))
    D = np.abs(D)
    outputs = _amp_to_db(_linear_to_mel(D))
    return _normalize(outputs)