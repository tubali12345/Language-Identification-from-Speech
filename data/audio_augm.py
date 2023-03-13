import numpy as np
import pyrubberband as pyrb
from scipy.signal import lfilter, butter


def butter_params(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_params(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def audio_augment(audio, sample_rate):
    # pitch shift
    if np.random.random() < 0.2:
        pitch = np.random.uniform(-1, 1)
        audio = pyrb.pitch_shift(audio, sample_rate, pitch)

    # volume change
    if np.random.random() < 0.2:
        modulation = np.random.uniform(0.5, 1.5)
        audio = audio * modulation

    # roll
    if np.random.random() < 0.2:
        shift = int(np.random.uniform(0, len(audio) / 2))
        audio = np.roll(audio, shift)

    # telephone simulation
    if np.random.random() < 0.1:
        low_freq = np.random.randint(100, 400) + 0.
        high_freq = np.random.randint(2000, 7000) + 0.
        audio = butter_bandpass_filter(audio, low_freq, high_freq, sample_rate, order=6)
    return audio
