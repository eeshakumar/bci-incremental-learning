import matplotlib.pyplot as plt


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

