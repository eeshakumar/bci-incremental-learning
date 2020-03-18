import sklearn.model_selection
import numpy as np


def normal(features, labels, test_ratio, random_state):
    """
    Split a dataset into training and test parts. Simple wrapper for train_test_split.
    """
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=test_ratio, random_state=random_state, stratify=labels
    )
    return x_train, x_test, y_train, y_test


def incremental_split(train_features, train_labels, bulk_batch_ratio=0.3):
    """
    Splits a dataset into batches according to the specified ratios.
    The idea is to first train on bulk_batch_ratio sized training examples and follow through with
    incrementally training on subsequent batches by per sample.
    """
    x_train_bulk, x_train_incremental, y_train_bulk, y_train_incremental = sklearn.model_selection\
        .train_test_split(train_features, train_labels, train_size=bulk_batch_ratio)
    return x_train_bulk, x_train_incremental, y_train_bulk, y_train_incremental


def k_way_split(train_features, train_labels, k=3, bulk_ratio=0.5):
    """
    This function just splits the data into 3 parts,
    with the first part being the largest chunk of half the size of dataset.
    Bulk ratio gives the size of the first largest chunk while the remaining data is equally split.
    """
    total_idx = train_labels.shape[0]
    first_split_idx = int(0.5 * total_idx)
    second_split_idx = first_split_idx + int(bulk_ratio * total_idx / 2)
    ds_features = list()
    ds_labels = list()
    s1_features, s2_features, s3_features = np.split(train_features, [first_split_idx, second_split_idx, total_idx])[0:3]
    s1_labels, s2_labels, s3_labels = np.split(train_labels, [first_split_idx, second_split_idx, total_idx])[0:3]
    ds_features.append(s1_features)
    ds_features.append(s2_features)
    ds_features.append(s3_features)
    ds_labels.append(s1_labels)
    ds_labels.append(s2_labels)
    ds_labels.append(s3_labels)
    return ds_features, ds_labels




