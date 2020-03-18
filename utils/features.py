import numpy as np
from utils.dataset import Dataset


def _norm_min_max(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def _norm_mean_std(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev


def normalize(data, normalization_type):
    """Normalize data.

    Normalize according to mean-std or min-max

    """
    norm_fns = {"mean_std": _norm_mean_std, "min_max": _norm_min_max}
    if not normalization_type in norm_fns:
        raise Exception("Normalization method '{m}' is not supported".format(m=normalization_type))
    if isinstance(data, Dataset):
        return norm_fns[normalization_type](data.raw_data)
    else:
        return norm_fns[normalization_type](data)
