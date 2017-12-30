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
    wav = librosa.core.load(wav_file)  # load as floating point time series


    pass


def _pre_emphasis(inputs):
    return signal.lfilter([1, -hp.pre_emphasis], [1], inputs)

def _stft(inputs):
    hop_length = int(hp.frame_shift/1000 * hp.sample_rate)      # frame shift(second) * sample rate(HZ/second)
    win_length = int(hp.frame_length/1000 * hp.sample_rate)     # frame length(second) * sample rate(HZ/second)
    return librosa.stft(inputs, n_fft=hp.num_freq, hop_length=hop_length, win_length=win_length, window='hann')

def _amp_to_db(inputs):
    return 20 * np.log10(np.maximum(1e-5, inputs))      # [1e-6, 1e-7, 10, 5] -> [1e-5, 1e-5, 10, 5]

def _normalize(inputs):
    pass


def _get_spectrogram(inputs):
    D = _stft(_pre_emphasis(inputs))
    D = np.abs(D)
    outputs = _amp_to_db(D) - hp.amp_reference
    return _normalize(outputs)