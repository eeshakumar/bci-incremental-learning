from .dataset import Dataset
import numpy as np


def get_values(data, btr_data):
    fs = data.sampling_freq
    labels = data.labels
    trial_len = data.trial_len
    trial_offset = data.trial_offset
    trials = data.trials
    if btr_data is not None:
        sample_data = btr_data
    else:
        sample_data = data.eeg_data
    return fs, labels, trial_len, trial_offset, trials, sample_data


def init_raw_electrode_reading(class_trials, trial_len, trial_offset, fs):
    raw_data = np.zeros((len(class_trials),
                        fs * (trial_len + trial_offset)))
    return raw_data, raw_data.copy(), raw_data.copy()


def get_class_labels(trials, labels, value):
    class_trials =  trials[np.where(labels == value)[0]]
    return class_trials


def init_raw_data(sample_data, trial_class_idx, trial_offset, fs, trial_len, electrode_idx):
    return sample_data[trial_class_idx - (trial_offset * fs): trial_class_idx + (trial_len * fs), electrode_idx]


def extract_trials_2class(data):

    fs, labels, trial_len, trial_offset, trials, sample_data = get_values(data)

    # 1-> Left, 2-> Right
    c1_trials = get_class_labels(trials, labels, 0)
    c2_trials = get_class_labels(trials, labels, 1)

    raw_c3_c1, raw_cz_c1, raw_c4_c1 = init_raw_electrode_reading(c1_trials, trial_len, trial_offset, fs)
    raw_c3_c2, raw_cz_c2, raw_c4_c2 = init_raw_electrode_reading(c2_trials, trial_len, trial_offset, fs)

    for i, (idx_c1, idx_c2) in enumerate(zip(c1_trials, c2_trials)):
        raw_c3_c1[i, :] = init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 0)
        raw_cz_c1[i, :] = init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 1)
        raw_c4_c1[i, :] = init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 2)

        raw_c3_c2[i, :] = init_raw_data(sample_data, idx_c2, trial_offset, fs, trial_len, 0)
        raw_cz_c2[i, :] = init_raw_data(sample_data, idx_c2, trial_offset, fs, trial_len, 1)
        raw_c4_c2[i, :] = init_raw_data(sample_data, idx_c2, trial_offset, fs, trial_len, 2)

    return np.array((raw_c3_c1, raw_cz_c1, raw_c4_c1, raw_c3_c2, raw_cz_c2, raw_c4_c2))


def extract_trials_4class(data, btr_data):

    fs, labels, trial_len, trial_offset, trials, sample_data = get_values(data, btr_data)

    # 1-> Left Hand, 2-> Right Hand, 3-> Both Foot(down) 4-> Tongue(up)
    c1_trials = get_class_labels(trials, labels, 0)
    c2_trials = get_class_labels(trials, labels, 1)
    c3_trials = get_class_labels(trials, labels, 2)
    c4_trials = get_class_labels(trials, labels, 3)

    raw_c3_c1, raw_cz_c1, raw_c4_c1 = init_raw_electrode_reading(c1_trials, trial_len, trial_offset, fs)
    raw_c3_c2, raw_cz_c2, raw_c4_c2 = init_raw_electrode_reading(c2_trials, trial_len, trial_offset, fs)
    raw_c3_c3, raw_cz_c3, raw_c4_c3 = init_raw_electrode_reading(c3_trials, trial_len, trial_offset, fs)
    raw_c3_c4, raw_cz_c4, raw_c4_c4 = init_raw_electrode_reading(c4_trials, trial_len, trial_offset, fs)
    for i, (idx_c1, idx_c2, idx_c3, idx_c4) in enumerate(zip(c1_trials, c2_trials, c3_trials, c4_trials)):

        #TODO: Figure a long term solution.
        # At some point, the data is incomplete with shape 1977 instead of 2000. Kind of a hack.
        if init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 0).shape[0] == \
                raw_c3_c1[i, :].shape[0]:

            raw_c3_c1[i, :] = init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 0)
            raw_cz_c1[i, :] = init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 1)
            raw_c4_c1[i, :] = init_raw_data(sample_data, idx_c1, trial_offset, fs, trial_len, 2)

            raw_c3_c2[i, :] = init_raw_data(sample_data, idx_c2, trial_offset, fs, trial_len, 0)
            raw_cz_c2[i, :] = init_raw_data(sample_data, idx_c2, trial_offset, fs, trial_len, 1)
            raw_c4_c2[i, :] = init_raw_data(sample_data, idx_c2, trial_offset, fs, trial_len, 2)

            raw_c3_c3[i, :] = init_raw_data(sample_data, idx_c3, trial_offset, fs, trial_len, 0)
            raw_cz_c3[i, :] = init_raw_data(sample_data, idx_c3, trial_offset, fs, trial_len, 1)
            raw_c4_c3[i, :] = init_raw_data(sample_data, idx_c3, trial_offset, fs, trial_len, 2)

            raw_c3_c4[i, :] = init_raw_data(sample_data, idx_c4, trial_offset, fs, trial_len, 0)
            raw_cz_c4[i, :] = init_raw_data(sample_data, idx_c4, trial_offset, fs, trial_len, 1)
            raw_c4_c4[i, :] = init_raw_data(sample_data, idx_c4, trial_offset, fs, trial_len, 2)

    return np.array((raw_c3_c1, raw_cz_c1, raw_c4_c1,
                     raw_c3_c2, raw_cz_c2, raw_c4_c2,
                     raw_c3_c3, raw_cz_c3, raw_c4_c3,
                     raw_c3_c4, raw_cz_c4, raw_c4_c4,))
