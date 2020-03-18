from .dataset import Dataset
import numpy as np


def extract_trials_2class(data, filtered=None, trials=None, labels=None, sampling_freq=0, trial_len=9, trial_offset=0):

    if isinstance(data, Dataset) or (filtered is not None):
        fs = data.sampling_freq
        labels = data.labels
        trial_len = data.trial_len
        trial_offset = data.trial_offset
        trials = data.trials

        if filtered is None:
            sample_data = data.eeg_data
        else:
            sample_data = filtered
    else:
        sample_data = data
        fs = sampling_freq
        trial_len = trial_len
        trial_offset = trial_offset

    # 1-> Left, 2-> Right
    c1_trials = trials[np.where(labels == 0)[0]]
    c2_trials = trials[np.where(labels == 1)[0]]

    raw_c3_c1 = np.zeros((len(c1_trials),
                          fs * (trial_len + trial_offset)))
    raw_cz_c1 = raw_c3_c1.copy()
    raw_c4_c1 = raw_c3_c1.copy()

    raw_c3_c2 = np.zeros((len(c2_trials),
                          fs * (trial_len + trial_offset)))
    raw_cz_c2 = raw_c3_c2.copy()
    raw_c4_c2 = raw_c3_c2.copy()

    for i, (idx_c1, idx_c2) in enumerate(zip(c1_trials, c2_trials)):
        raw_c3_c1[i, :] = sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 0]
        raw_cz_c1[i, :] = sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 1]
        raw_c4_c1[i, :] = sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 2]

        raw_c3_c2[i, :] = sample_data[idx_c2 - (trial_offset * fs) : idx_c2 + (trial_len * fs), 0]
        raw_cz_c2[i, :] = sample_data[idx_c2 - (trial_offset * fs) : idx_c2 + (trial_len * fs), 1]
        raw_c4_c2[i, :] = sample_data[idx_c2 - (trial_offset * fs): idx_c2 + (trial_len * fs), 2]

    return np.array((raw_c3_c1, raw_cz_c1, raw_c4_c1, raw_c3_c2, raw_cz_c2, raw_c4_c2))


def extract_trials_4class():
    pass
