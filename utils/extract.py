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


def extract_trials_4class(data, filtered=None, trials=None, labels=None, sampling_freq=0, trial_len=8, trial_offset=0):

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

    # 1-> Left Hand, 2-> Right Hand, 3-> Both Foot(down) 4-> Tongue(up)
    c1_trials = trials[np.where(labels == 0)[0]]
    c2_trials = trials[np.where(labels == 1)[0]]
    c3_trials = trials[np.where(labels == 2)[0]]
    c4_trials = trials[np.where(labels == 3)[0]]
    print(c1_trials.shape, c2_trials.shape, c3_trials.shape, c4_trials.shape)

    raw_c3_c1 = np.zeros((len(c1_trials),
                          fs * (trial_len + trial_offset)))
    raw_cz_c1 = raw_c3_c1.copy()
    raw_c4_c1 = raw_c3_c1.copy()

    print(raw_c4_c1.shape)

    raw_c3_c2 = np.zeros((len(c2_trials),
                          fs * (trial_len + trial_offset)))
    raw_cz_c2 = raw_c3_c2.copy()
    raw_c4_c2 = raw_c3_c2.copy()

    raw_c3_c3 = np.zeros((len(c3_trials),
                          fs * (trial_len + trial_offset)))
    raw_cz_c3 = raw_c3_c3.copy()
    raw_c4_c3 = raw_c3_c3.copy()

    raw_c3_c4 = np.zeros((len(c4_trials),
                          fs * (trial_len + trial_offset)))
    raw_cz_c4 = raw_c3_c4.copy()
    raw_c4_c4 = raw_c3_c4.copy()

    for i, (idx_c1, idx_c2, idx_c3, idx_c4) in enumerate(zip(c1_trials, c2_trials, c3_trials, c4_trials)):

        if(sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 0].shape[0] == raw_c3_c1[i, :].shape[0]):

            raw_c3_c1[i, :] = sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 0]
            raw_cz_c1[i, :] = sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 1]
            raw_c4_c1[i, :] = sample_data[idx_c1 - (trial_offset * fs) : idx_c1 + (trial_len * fs), 2]

            raw_c3_c2[i, :] = sample_data[idx_c2 - (trial_offset * fs) : idx_c2 + (trial_len * fs), 0]
            raw_cz_c2[i, :] = sample_data[idx_c2 - (trial_offset * fs) : idx_c2 + (trial_len * fs), 1]
            raw_c4_c2[i, :] = sample_data[idx_c2 - (trial_offset * fs): idx_c2 + (trial_len * fs), 2]

            raw_c3_c3[i, :] = sample_data[idx_c3 - (trial_offset * fs) : idx_c3 + (trial_len * fs), 0]
            raw_cz_c3[i, :] = sample_data[idx_c3 - (trial_offset * fs) : idx_c3 + (trial_len * fs), 1]
            raw_c4_c3[i, :] = sample_data[idx_c3 - (trial_offset * fs): idx_c3 + (trial_len * fs), 2]

            raw_c3_c4[i, :] = sample_data[idx_c4 - (trial_offset * fs) : idx_c4 + (trial_len * fs), 0]
            raw_cz_c4[i, :] = sample_data[idx_c4 - (trial_offset * fs) : idx_c4 + (trial_len * fs), 1]
            raw_c4_c4[i, :] = sample_data[idx_c4 - (trial_offset * fs): idx_c4 + (trial_len * fs), 2]

    return np.array((raw_c3_c1, raw_cz_c1, raw_c4_c1,
                     raw_c3_c2, raw_cz_c2, raw_c4_c2,
                     raw_c3_c3, raw_cz_c3, raw_c4_c3,
                     raw_c3_c4, raw_cz_c4, raw_c4_c4,))
