from abc import ABC, abstractmethod
import numpy as np


class DatasetError(Exception):
    pass


class Dataset(ABC):
    """
    Base class.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load(self, **kwargs):
        """
        Load data depending on type of dataset. To be expanded in derived classes.
        """
        return self

    def print_stats(self):
        """
        Print dataset specs
        """
        print("EEG data shape :", self.eeg_data.shape)
        print("Labels shape :", self.labels.shape)
        print("Trials shape :", self.trials.shape)
        print("Trial Length :", self.trial_len)
        print("Sampling frequency :", self.sampling_freq)
        print("Cue Interval :", self.cue_interval)
        print("Number of classes :", np.unique(self.labels))
