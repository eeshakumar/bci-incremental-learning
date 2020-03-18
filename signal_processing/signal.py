from utils.dataset import Dataset

import numpy as np
import scipy.signal
import scipy.stats


class ButterBandpass:
    """
    Filter class for a Butterworth bandpass filter.
    """

    def __init__(self, lowcut, highcut, order=2, fs=512):
        """
        Initialize the Butterworth bandpass filter.
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = scipy.signal.butter(order, [low, high], btype="bandpass")

    def process(self, data, axis=0):
        """
        Apply filter along axis
        """
        return scipy.signal.filtfilt(self.b, self.a, data, axis)


def butter_bandpass(data, lo, hi, axis=0, **kwargs):
    """
    Apply bandpass filter.
    """
    if isinstance(data, Dataset):
        flt = ButterBandpass(lo, hi, fs=data.sampling_freq, **kwargs)
        filtered = [flt.process(data.eeg_data[:, i], axis) for i in range(data.eeg_data.shape[1])]
        reshaped = [f.reshape(-1, 1) for f in filtered]
        return np.hstack(reshaped)
    else:
        flt = ButterBandpass(lo, hi, **kwargs)
        return flt.process(data, axis)
