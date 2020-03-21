import matplotlib.pyplot as plt
import numpy as np

def plot_xy(data, title="", label=""):
    """
    Plot 1D data.
    """
    assert(data.shape[1] == 1)
    data_range = list(range(data.shape[0]))
    data = list(data.flatten())
    if label:
        plt.plot(data_range, data, linestyle='-', marker='o', color='g', label=label)
        plt.legend()
    else:
        plt.plot(data_range, data, linestyle='-', marker='o', color='g')
    plt.title(title)
    plt.show()


def plot_brain_signals(data, btr_data, electrode_idx, title=""):
    """
    Plot filtered vs. unfiltered eeg data
    """
    plt.plot(data.eeg_data[data.trials[0]: data.trials[1], electrode_idx], alpha=0.6, label='unfiltered')
    plt.plot(btr_data[data.trials[0]: data.trials[1], electrode_idx], alpha=0.6, label='filtered')
    plt.legend()
    plt.title(title)
    plt.show()


def plot_spectral_representation(data_class, class_idx, sampling_freq=250, title='Spectrogram', axis=1,
                                 NFFT=128, noverlap=100):
    """
    Plot spectogram of EEG data by class.
    """
    plot_data=np.mean(data_class[class_idx], axis=axis)
    plt.specgram(plot_data, NFFT=NFFT, Fs=sampling_freq, noverlap=noverlap, cmap=plt.get_cmap('Spectral'))
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.colorbar()
    plt.show()