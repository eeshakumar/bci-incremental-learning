from .dataset import Dataset, DatasetError
import os
import numpy as np
import scipy.io


class Graz(Dataset):
    """
    Graz dataset from BCNI2020 competition.
    http://bnci-horizon-2020.eu/database/data-sets
    """

    def __init__(self, base_dir, identifier, trail_len, cue_interval, trial_offset, expected_freq, **kwargs):
        """
        Init Graz data specifics. trial_len, cue_interval, cue_offset and
        expected_freq are expected depending on supporting literature.
        """

        super(Graz, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = base_dir

        self.trial_len = trail_len
        self.cue_interval = cue_interval
        self.trial_offset = trial_offset
        self.expected_freq = expected_freq

        # the graz dataset is split into (T)raining and (E)valuation files
        self.matT = os.path.join(self.data_dir, "{id}T.mat".format(id=self.data_id))
        self.matE = os.path.join(self.data_dir, "{id}E.mat".format(id=self.data_id))

        for f in [self.matT, self.matE]:
            if not os.path.isfile(f):
                raise DatasetError(
                    "Graz Dataset ({id}) file '{f}' unavailable".format(id=self.data_id, f=f)
                )

        # variables to store data
        self.eeg_data = None
        self.labels = None
        self.trials = None
        self.sampling_freq = None

    def load(self, **kwargs):
        """
        Load a dataset. The array from mat1, mat2 can be printed to figure values.

        """
        mat1 = scipy.io.loadmat(self.matT)["data"]
        mat2 = scipy.io.loadmat(self.matE)["data"]

        # EEG Data 0 C3 1 Cz 2 C4
        data_bt = []
        labels_bt = []
        trials_bt = []
        n_experiments = 3
        for i in range(n_experiments):

            data = mat1[0, i][0][0][0]
            trials = mat1[0, i][0][0][1]
            labels = mat1[0, i][0][0][2] - 1
            fs = mat1[0, i][0][0][3].flatten()[0]
            if fs != self.expected_freq:
                raise DatasetError(
                    "Frequency mismatch (expected {}, got {})".format(
                        self.expected_freq, fs
                    )
                )
            data_bt.append(data)
            labels_bt.append(labels)
            trials_bt.append(trials)

        trials_bt[1] += data_bt[0].shape[0]
        trials_bt[2] += data_bt[0].shape[0] + data_bt[1].shape[0]

        data_bt = np.concatenate((data_bt[0], data_bt[1], data_bt[2]))
        trials_bt = np.concatenate((trials_bt[0], trials_bt[1], trials_bt[2]))
        labels_bt = np.concatenate((labels_bt[0], labels_bt[1], labels_bt[2]))

        self.eeg_data = data_bt[:, :3]
        self.trials = trials_bt
        self.trials = self.trials.ravel()
        self.labels = labels_bt
        self.labels = self.labels.ravel()
        self.sampling_freq = self.expected_freq

        return self


class GrazA(Dataset):
    """
    Class to hold 4class motor imagery data from Graz.
    """

    def __init__(self, base_dir, identifier, trail_len, cue_interval, trial_offset, expected_freq, **kwargs):
        """
        Init Graz data specifics. trial_len, cue_interval, cue_offset and
        expected_freq are expected depending on supporting literature.
        """

        super(GrazA, self).__init__(**kwargs)

        self.base_dir = base_dir
        self.data_id = identifier
        self.data_dir = base_dir

        self.trial_len = trail_len
        self.cue_interval = cue_interval
        self.trial_offset = trial_offset
        self.expected_freq = expected_freq

        # the graz dataset is split into (T)raining and (E)valuation files
        self.matT = os.path.join(self.data_dir, "{id}T.mat".format(id=self.data_id))
        self.matE = os.path.join(self.data_dir, "{id}E.mat".format(id=self.data_id))

        for f in [self.matT, self.matE]:
            if not os.path.isfile(f):
                raise DatasetError(
                    "Graz Dataset ({id}) file '{f}' unavailable".format(id=self.data_id, f=f)
                )

        # variables to store data
        self.eeg_data = None
        self.labels = None
        self.trials = None
        self.sampling_freq = None

    def load(self, **kwargs):
        """
        Load a dataset. The array from mat1, mat2 can be printed to figure values.

        """
        mat1 = scipy.io.loadmat(self.matT)["data"]
        mat2 = scipy.io.loadmat(self.matE)["data"]

        # EEG Data 0 C3 1 Cz 2 C4
        data_bt = []
        labels_bt = []
        trials_bt = []
        n_experiments = 9
        # Trials 0-3 were empty A01
        for i in range(4, n_experiments):

            data = mat1[0, i][0][0][0]
            trials = mat1[0, i][0][0][1]
            labels = mat1[0, i][0][0][2] - 1
            fs = mat1[0, i][0][0][3].flatten()[0]
            if fs != self.expected_freq:
                raise DatasetError(
                    "Frequency mismatch (expected {}, got {})".format(
                        self.expected_freq, fs
                    )
                )
            data_bt.append(data)
            labels_bt.append(labels)
            trials_bt.append(trials)

        trials_bt[1] += data_bt[0].shape[0]
        trials_bt[2] += data_bt[0].shape[0] + data_bt[1].shape[0]
        trials_bt[3] += data_bt[0].shape[0] + data_bt[1].shape[0] + data_bt[2].shape[0]
        trials_bt[4] += data_bt[0].shape[0] + data_bt[1].shape[0] + data_bt[2].shape[0] + data_bt[3].shape[0]

        data_bt = np.concatenate((data_bt[0], data_bt[1], data_bt[2], data_bt[3], data_bt[4]))
        trials_bt = np.concatenate((trials_bt[0], trials_bt[1], trials_bt[2], trials_bt[3], trials_bt[4]))
        labels_bt = np.concatenate((labels_bt[0], labels_bt[1], labels_bt[2], labels_bt[3], labels_bt[4]))

        self.eeg_data = data_bt[:, :5]
        self.trials = trials_bt
        self.trials = self.trials.ravel()
        self.labels = labels_bt
        self.labels = self.labels.ravel()
        self.sampling_freq = self.expected_freq

        return self