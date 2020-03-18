"""Signal processing utilities, filters, and data post-processing routines.


Every filter comes in form of a pair:
1) filter class
2) filter commodity function

The commodity functions internally create a filter class and invoke the
corresponding ``process`` method.  Often, however, usage requires to apply a
filter multiple times. In this case, the filter classes should be used directly
as this avoids redundant initialization of the filters.

If possible, the filters are initialized with arguments that were found to be
suitable for most EEG/EMG post-processing needs. Other arguments need to be
passed when creating a filter class. The commodity functions forward all
(unknown) arguments to the filter initialization.

"""

# TODO: description above. check if we really have a filter class for every
#       filter, or if we specify them

from .data.dataset import Dataset

import numpy as np
import scipy.signal
from scipy import signal
import scipy.stats
import sklearn.decomposition
import pywt
from sklearn.preprocessing import MinMaxScaler


class ButterBandpass:
    """Filter class for a Butterworth bandpass filter.

    """

    def __init__(self, lowcut, highcut, order=2, fs=512):
        """Initialize the Butterworth bandpass filter.

        Args:
            lowcut (float): low cut-off frequency
            highcut (float): high cut-off frequency
            order (int): order of the Butterworth bandpass filter
            fs (int): sampling frequency

        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = scipy.signal.butter(order, [low, high], btype="bandpass")

    def process(self, data, axis=0):
        """Apply the filter to data along a given axis.

        Args:
            data (array_like): data to filter
            axis (int): along which data to filter

        Returns:
            ndarray: Result of the same shape as data

        """
        return scipy.signal.filtfilt(self.b, self.a, data, axis)


def butter_bandpass(data, lo, hi, axis=0, **kwargs):
    """Apply a Butterworth bandpass filter to some data.

    The function either takes an ``array_like`` object (e.g. numpy's ndarray) or
    an instance of a gumpy.data.Dataset subclass as first argument.

    Args:
        data (array_like or Dataset instance): input data. If this is an
            instance of a Dataset subclass, the sampling frequency will be extracted
            automatically.
        lo (float): low cutoff frequency.
        hi (float): high cutoff frequency.
        axis (int): along which axis of data the filter should be applied. Default = 0.
        **kwargs: Additional keyword arguments that will be passed to ``gumpy.signal.ButterBandstop``.

    Returns:
        array_like: data filtered long the specified axis.

    """
    if isinstance(data, Dataset):
        flt = ButterBandpass(lo, hi, fs=data.sampling_freq, **kwargs)
        filtered = [flt.process(data.raw_data[:, i], axis) for i in range(data.raw_data.shape[1])]
        reshaped = [f.reshape(-1, 1) for f in filtered]
        return np.hstack(reshaped)
    else:
        flt = ButterBandpass(lo, hi, **kwargs)
        return flt.process(data, axis)


class ButterHighpass:
    """Filter class for a Butterworth bandpass filter.

    """

    def __init__(self, cutoff, order=4, fs=512):
        """Initialize the Butterworth highpass filter.

        Args:
            cutoff (float): cut-off frequency
            order (int): order of the Butterworth bandpass filter
            fs (int): sampling frequency

        """
        self.cutoff = cutoff
        self.order = order

        nyq = 0.5 * fs
        high = cutoff / nyq
        self.b, self.a = scipy.signal.butter(order, high, btype="highpass")

    def process(self, data, axis=0):
        """Apply the filter to data along a given axis.

        Args:
            data (array_like): data to filter
            axis (int): along which data to filter

        Returns:
            ndarray: Result of the same shape as data

        """
        return scipy.signal.filtfilt(self.b, self.a, data, axis)


def butter_highpass(data, cutoff, axis=0, **kwargs):
    """Apply a Butterworth highpass filter to some data.

    The function either takes an ``array_like`` object (e.g. numpy's ndarray) or
    an instance of a gumpy.data.Dataset subclass as first argument.

    Args:
        data (array_like or Dataset instance): input data. If this is an
            instance of a Dataset subclass, the sampling frequency will be extracted
            automatically.
        cutoff (float): cutoff frequency.
        axis (int): along which axis of data the filter should be applied. Default = 0.
        **kwargs: Additional keyword arguments that will be passed to ``gumpy.signal.ButterBandstop``.

    Returns:
        array_like: data filtered long the specified axis.

    """

    if isinstance(data, Dataset):
        flt = ButterHighpass(cutoff, fs=data.sampling_freq, **kwargs)
        filtered = [flt.process(data.raw_data[:, i], axis) for i in range(data.raw_data.shape[1])]
        reshaped = [f.reshape(-1, 1) for f in filtered]
        return np.hstack(reshaped)
    else:
        flt = ButterHighpass(cutoff, **kwargs)
        return flt.process(data, axis)


class ButterLowpass:
    """Filter class for a Butterworth lowpass filter.

    """

    def __init__(self, cutoff, order=4, fs=512):
        """Initialize the Butterworth lowpass filter.

        Args:
            cutoff (float): cut-off frequency
            order (int): order of the Butterworth bandpass filter
            fs (int): sampling frequency

        """

        self.cutoff = cutoff
        self.order = order

        nyq = 0.5 * fs
        low = cutoff / nyq
        self.b, self.a = scipy.signal.butter(order, low, btype="lowpass")

    def process(self, data, axis=0):
        """Apply the filter to data along a given axis.

        Args:
            data (array_like): data to filter
            axis (int): along which data to filter

        Returns:
            ndarray: Result of the same shape as data

        """
        return scipy.signal.filtfilt(self.b, self.a, data, axis)


def butter_lowpass(data, cutoff, axis=0, **kwargs):
    """Apply a Butterworth lowpass filter to some data.

    The function either takes an ``array_like`` object (e.g. numpy's ndarray) or
    an instance of a gumpy.data.Dataset subclass as first argument.

    Args:
        data (array_like or Dataset instance): input data. If this is an
            instance of a Dataset subclass, the sampling frequency will be extracted
            automatically.
        cutoff (float): cutoff frequency.
        axis (int): along which axis of data the filter should be applied. Default = 0.
        **kwargs: Additional keyword arguments that will be passed to ``gumpy.signal.ButterBandstop``.

    Returns:
        array_like: data filtered long the specified axis.

    """
    if isinstance(data, Dataset):
        flt = ButterLowpass(cutoff, fs=data.sampling_freq, **kwargs)
        filtered = [flt.process(data.raw_data[:, i], axis) for i in range(data.raw_data.shape[1])]
        reshaped = [f.reshape(-1, 1) for f in filtered]
        return np.hstack(reshaped)
    else:
        flt = ButterLowpass(cutoff, **kwargs)
        return flt.process(data, axis)


class ButterBandstop:
    """Filter class for a Butterworth bandstop filter.

    """

    def __init__(self, lowpass=49, highpass=51, order=4, fs=512):
        """Initialize the Butterworth bandstop filter.

        Args:
            lowpass (float): low cut-off frequency. Default = 49
            highapss (float): high cut-off frequency. Default = 51
            order (int): order of the Butterworth bandpass filter.
            fs (int): sampling frequency
        """
        self.lowpass = lowpass
        self.highpass = highpass
        self.order = order

        nyq = 0.5 * fs
        low = lowpass / nyq
        high = highpass / nyq
        self.b, self.a = scipy.signal.butter(order, [low, high], btype="bandstop")

    def process(self, data, axis=0):
        """Apply the filter to data along a given axis.

        Args:
            data (array_like): data to filter
            axis (int): along which data to filter

        Returns:
            ndarray: Result of the same shape as data

        """
        return scipy.signal.filtfilt(self.b, self.a, data, axis)


def butter_bandstop(data, axis=0, **kwargs):
    """Apply a Butterworth bandstop filter to some data.

    The function either takes an ``array_like`` object (e.g. numpy's ndarray) or
    an instance of a gumpy.data.Dataset subclass as first argument.

    Args:
        data (array_like or Dataset instance): input data. If this is an
            instance of a Dataset subclass, the sampling frequency will be extracted
            automatically.
        axis (int): along which axis of data the filter should be applied. Default = 0.
        **kwargs: Additional keyword arguments that will be passed to ``gumpy.signal.ButterBandstop``.

    Returns:
        array_like: data filtered long the specified axis.

    """
    if isinstance(data, Dataset):
        flt = ButterBandstop(lo, hi, fs=data.sampling_freq, **kwargs)
        filtered = [flt.process(data.raw_data[:, i], axis) for i in range(data.raw_data.shape[1])]
        reshaped = [f.reshape(-1, 1) for f in filtered]
        return np.hstack(reshaped)
    else:
        flt = ButterBandstop(lo, hi, **kwargs)
        return flt.process(data, axis)


class Notch:
    """Filter class for a notch filter.

    """

    def __init__(self, cutoff=50, Q=30, fs=512):
        """Initialize the notch filter.

        Args:
            cutoff (float): cut-off frequency. Default = 50.
            Q (float): Quality factor. Default = 30.
            fs (int): sampling frequency. Default = 256
        """
        self.cutoff = cutoff
        self.Q = Q

        nyq = 0.5 * fs
        w0 = cutoff / nyq
        self.b, self.a = scipy.signal.iirnotch(w0, Q)

    def process(self, data, axis=0):
        """Apply the filter to data along a given axis.

        Args:
            data (array_like): data to filter
            axis (int): along which data to filter

        Returns:
            ndarray: Result of the same shape as data

        """
        return scipy.signal.filtfilt(self.b, self.a, data, axis)


def notch(data, cutoff, axis=0, **kwargs):
    """Apply a notch filter to data.

    The function either takes an ``array_like`` object (e.g. numpy's ndarray) or
    an instance of a gumpy.data.Dataset subclass as first argument.

    Args:
        data (array_like or Dataset instance): input data.
        cutoff (float): cutoff frequency. Default = 50.
        axis (int): along which axis of data the filter should be applied. Default = 0.
        Q (float): quality factor. Default = 30.
        fs (int): sampling frequenct. Default = 256.

    Returns:
        array_like: data filtered long the specified axis.

    """
    if isinstance(data, Dataset):
        flt = Notch(cutoff, fs=data.sampling_freq, **kwargs)
        filtered = [flt.process(data.raw_data[:, i], axis) for i in range(data.raw_data.shape[1])]
        reshaped = [f.reshape(-1, 1) for f in filtered]
        return np.hstack(reshaped)
    else:
        flt = Notch(cutoff, **kwargs)
        return flt.process(data, axis)


def _norm_min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def data_scaler(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def _norm_mean_std(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev


def normalize(data, normalization_type):
    """Normalize data.

    Normalize data either by shifting and rescaling the data to [0,1]
    (``min_max``) or by rescaling via mean and standard deviation
    (``mean_std``).

    Args:
        data (array_like): Input data
        normalization_type (str): One of ``mean_std``, ``mean_std``

    Returns:
        ndarray: normalized data with same shape as ``data``

    Raises:
        Exception: if the normalization type is unknown.

    """
    norm_fns = {"mean_std": _norm_mean_std, "min_max": _norm_min_max}
    if not normalization_type in norm_fns:
        raise Exception("Normalization method '{m}' is not supported".format(m=normalization_type))
    if isinstance(data, Dataset):
        return norm_fns[normalization_type](data.raw_data)
    else:
        return norm_fns[normalization_type](data)


def dwt(raw_eeg_data, level, **kwargs):
    """Multilevel Discrete Wavelet Transform (DWT).

    Compute the DWT for a raw eeg signal on multiple levels.

    Args:
        raw_eeg_data (array_like): input data
        level (int >= 0): decomposition levels
        **kwargs: Additional arguments that will be forwarded to ``pywt.wavedec``

    Returns:
        A 2-element tuple containing

        - **float**: mean value of the first decomposition coefficients
        - **list**: list of mean values for the individual (detail) decomposition coefficients

    """
    wt_coeffs = pywt.wavedec(data=raw_eeg_data, level=level, **kwargs)

    # A7:  0 Hz - 1 Hz
    cAL_mean = np.nanmean(wt_coeffs[0], axis=0)
    details = []

    # For Fs = 128 H
    for i in range(1, level + 1):
        # D7:  1 Hz - 2 Hz
        cDL_mean = np.nanmean(wt_coeffs[i], axis=0)
        details.append(cDL_mean)

    return cAL_mean, details


def correlation(x, y):
    """Compute the correlation between x and y using Pearson's r.

    """
    return scipy.stats.pearsonr(x, y)


def fft(data, fs):
    n = data.shape[-1]
    window = np.hanning(n)
    windowed = data * window
    spectrum = np.fft.fft(windowed)
    freq = np.fft.fftfreq(n, 1 / fs)
    half_n = np.ceil(n / 2)
    spectrum_half = (2 / n) * spectrum[..., :half_n]
    freq_half = freq[:half_n]
    return freq_half, np.abs(spectrum_half)


def fft_process(data):
    f = np.fft.fft2(data)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift))


def continuous_wavelet(X):
    cwt = None
    cwt = []
    for k in X:
        widths = np.arange(1, 2)
        sig_ch1 = k[:, 0]
        cwt_ch1 = signal.cwt(sig_ch1, signal.ricker, widths).T
        sig_ch2 = k[:, 1]
        cwt_ch2 = signal.cwt(sig_ch2, signal.ricker, widths).T
        sig_ch3 = k[:, 2]
        cwt_ch3 = signal.cwt(sig_ch3, signal.ricker, widths).T
        sig_ch4 = k[:, 3]
        cwt_ch4 = signal.cwt(sig_ch4, signal.ricker, widths).T
        sig_ch5 = k[:, 4]
        cwt_ch5 = signal.cwt(sig_ch5, signal.ricker, widths).T
        sig_ch6 = k[:, 5]
        cwt_ch6 = signal.cwt(sig_ch6, signal.ricker, widths).T
        sig_ch7 = k[:, 6]
        cwt_ch7 = signal.cwt(sig_ch7, signal.ricker, widths).T
        cwt.append(np.column_stack((cwt_ch1, cwt_ch2, cwt_ch3, cwt_ch4, cwt_ch5, cwt_ch6, cwt_ch7)))
    cwt = np.asarray(cwt)
    return cwt


def artifact_removal(X, n_components=None, check_result=True):
    """Remove artifacts from data.

    The artifacts are detected via Independent Component Analysis (ICA) and
    subsequently removed. To plot the results, use
    :func:`gumpy.plot.artifact_removal`

    Args:
        X (array_like): Data to remove artifacts from
        n_components (int): Number of components for ICA. If None is passed, all will be used
        check_result (bool): Examine/test the ICA model by reverting the mixing.


    Returns:
        A 2-tuple containing

        - **ndarray**: The reconstructed signal without artifacts.
        - **ndarray**: The mixing matrix that wqas used by ICA.

    """

    ica = sklearn.decomposition.FastICA(n_components)
    S_reconst = ica.fit_transform(X)
    A_mixing = ica.mixing_
    if check_result:
        assert np.allclose(X, np.dot(S_reconst, A_mixing.T) + ica.mean_)

    return S_reconst, A_mixing


def sliding_window(data, labels, window_sz, n_hop, n_start=0, show_status=False):

    """

    input: (array) data : matrix to be processed

           (int)   window_sz : nb of samples to be used in the window

           (int)   n_hop : size of jump between windows

    output:(array) new_data : output matrix of size (None, window_sz, feature_dim)



    """

    flag = 0

    for sample in range(data.shape[0]):

        tmp = np.array(
            [
                data[sample, i : i + window_sz, :]
                for i in np.arange(n_start, data.shape[1] - window_sz + n_hop, n_hop)
            ]
        )

        tmp_lab = np.array(
            [labels[sample] for i in np.arange(n_start, data.shape[1] - window_sz + n_hop, n_hop)]
        )

        if sample % 100 == 0 and show_status == True:

            print("Sample " + str(sample) + "processed!\n")

        if flag == 0:

            new_data = tmp

            new_lab = tmp_lab

            flag = 1

        else:

            new_data = np.concatenate((new_data, tmp))

            new_lab = np.concatenate((new_lab, tmp_lab))

    return new_data, new_lab
