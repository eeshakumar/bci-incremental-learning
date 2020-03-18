from sklearn.preprocessing import PolynomialFeatures
from .classification.classifier import available_classifiers
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import numpy as np
import scipy.linalg as la

# import sklearn.cross_validation
# from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def sequential_feature_selector(
    features, labels, classifier, k_features, kfold, selection_type, plot=True, **kwargs
):
    """Sequential feature selection to reduce the number of features.

    The function reduces a d-dimensional feature space to a k-dimensional
    feature space by sequential feature selection. The features are selected
    using ``mlxtend.feature_selection.SequentialFeatureSelection`` which
    essentially selects or removes a feature from the d-dimensional input space
    until the preferred size is reached.

    The function will pass ``ftype='feature'`` and forward ``features`` on to a
    classifier's ``static_opts`` method.

    Args:
        features: The original d-dimensional feature space
        labels: corresponding labels
        classifier (str or object): The classifier which should be used for
            feature selection. This can be either a string (name of a classifier
            known to Finpy) or an instance of a classifier which adheres
            to the sklearn classifier interface.
        k_features (int): Number of features to select
        kfold (int): k-fold cross validation
        selection_type (str): One of ``SFS`` (Sequential Forward Selection),
            ``SBS`` (Sequential Backward Selection), ``SFFS`` (Sequential Forward
            Floating Selection), ``SBFS`` (Sequential Backward Floating Selection)
        plot (bool): Plot the results of the dimensinality reduction
        **kwargs: Additional keyword arguments that will be passed to the
            Classifier instantiation

    Returns:
        A 3-element tuple containing

        - **feature index**: Index of features in the remaining set
        - **cv_scores**: cross validation scores during classification
        - **algorithm**: Algorithm that was used for search

    """

    # retrieve the appropriate classifier
    if isinstance(classifier, str):
        if not (classifier in available_classifiers):
            raise ClassifierError("Unknown classifier {c}".format(c=classifier.__repr__()))

        kwopts = kwargs.pop("opts", dict())
        # opts = dict()

        # retrieve the options that we need to forward to the classifier
        # TODO: should we forward all arguments to sequential_feature_selector ?
        opts = available_classifiers[classifier].static_opts(
            "sequential_feature_selector", features=features
        )
        opts.update(kwopts)

        # XXX: now merged into the static_opts invocation. TODO: test
        # if classifier == 'SVM':
        #     opts['cross_validation'] = kwopts.pop('cross_validation', False)
        # elif classifier == 'RandomForest':
        #     opts['cross_validation'] = kwopts.pop('cross_validation', False)
        # elif classifier == 'MLP':
        #     # TODO: check if the dimensions are correct here
        #     opts['hidden_layer_sizes'] = (features.shape[1], features.shape[2])
        # get all additional entries for the options
        # opts.update(kwopts)

        # retrieve a classifier object
        classifier_obj = available_classifiers[classifier](**opts)

        # extract the backend classifier
        clf = classifier_obj.clf
    else:
        # if we received a classifier object we'll just use this one
        clf = classifier.clf

    if selection_type == "SFS":
        algorithm = "Sequential Forward Selection (SFS)"
        sfs = SFS(
            clf,
            k_features,
            forward=True,
            floating=False,
            verbose=2,
            scoring="accuracy",
            cv=kfold,
            n_jobs=-1,
        )

    elif selection_type == "SBS":
        algorithm = "Sequential Backward Selection (SBS)"
        sfs = SFS(
            clf,
            k_features,
            forward=False,
            floating=False,
            verbose=2,
            scoring="accuracy",
            cv=kfold,
            n_jobs=-1,
        )

    elif selection_type == "SFFS":
        algorithm = "Sequential Forward Floating Selection (SFFS)"
        sfs = SFS(
            clf,
            k_features,
            forward=True,
            floating=True,
            verbose=2,
            scoring="accuracy",
            cv=kfold,
            n_jobs=-1,
        )

    elif selection_type == "SBFS":
        algorithm = "Sequential Backward Floating Selection (SFFS)"
        sfs = SFS(
            clf,
            k_features,
            forward=True,
            floating=True,
            verbose=2,
            scoring="accuracy",
            cv=kfold,
            n_jobs=-1,
        )

    else:
        raise Exception("Unknown selection type '{}'".format(selection_type))

    pipe = make_pipeline(StandardScaler(), sfs)
    pipe.fit(features, labels)
    subsets = sfs.subsets_
    feature_idx = sfs.k_feature_idx_
    cv_scores = sfs.k_score_

    if plot:
        fig1 = plot_sfs(sfs.get_metric_dict(), kind="std_dev")
        plt.ylim([0.5, 1])
        plt.title(algorithm)
        plt.grid()
        plt.show()

    return feature_idx, cv_scores, algorithm, sfs, clf


# Zero crossings: it is the number of times the waveform crosses zero. This feature provides an approximate estimation of frequency domain properties. The threshold avoids counting zero crossings induced by noise.


def rms(signal, fs, window_size, window_shift):
    """Root Mean Square.

    Args:
        signal (array_like): TODO
        fs (int): Sampling frequency
        window_size: TODO
        window_shift: TODO

    Returns:
        TODO:
    """
    duration = len(signal) / fs
    n_features = int(duration / (window_size - window_shift))

    features = np.zeros(n_features)

    for i in range(n_features):
        idx1 = int((i * (window_size - window_shift)) * fs)
        idx2 = int(((i + 1) * window_size - i * window_shift) * fs)
        rms = np.sqrt(np.mean(np.square(signal[idx1:idx2])))
        features[i] = rms

    return features


def RMS_features_extraction(data, trial_list, window_size, window_shift):
    """
    Extract RMS features from data
    Args:
        data: 2D (time points, Channels)
        trial_list: list of the trials
        window_size: Size of the window for extracting features
        window_shift: size of the overalp
    Returns:
        The features matrix (trials, features)
    """
    if window_shift > window_size:
        raise ValueError("window_shift > window_size")

    fs = data.sampling_freq

    n_features = int(data.duration / (window_size - window_shift))

    X = np.zeros((len(trial_list), n_features * 4))

    t = 0
    for trial in trial_list:
        # x3 is the worst of all with 43.3% average performance
        x1 = rms(trial[0], fs, window_size, window_shift)
        x2 = rms(trial[1], fs, window_size, window_shift)
        x3 = rms(trial[2], fs, window_size, window_shift)
        x4 = rms(trial[3], fs, window_size, window_shift)
        x = np.concatenate((x1, x2, x3, x4))
        X[t, :] = np.array([x])
        t += 1
    return X


# MAV
def mav(segment):
    mav = np.mean(np.abs(segment))
    return mav


# rms
def RMS(segment):
    rms = np.sqrt(np.mean(np.power(segment, 2)))
    return rms


# var
def var(segment):
    var = np.var(segment)
    return var


# Simple square integral: it gives a measure of the energy of the EMG signal.
def ssi(segment):
    ssi = np.sum(np.abs(np.power(segment, 2)))
    return ssi


# Slope sign changes: it is similar to the zero crossings feature. It also provides information about the frequency content of the signal. It is calculated as follows
def ssc(segment):
    N = len(segment)
    ssc = 0
    for n in range(1, N - 1):
        if (segment[n] - segment[n - 1]) * (segment[n] - segment[n + 1]) >= 0.001:
            ssc = ssc + 1
    return ssc


def CSP(tasks):
    """This function extracts Common Spatial Pattern (CSP) features.
    Args:
        For N tasks, N arrays are passed to CSP each with dimensionality (# of
        trials of task N) x (feature vector)
    Returns:
        A 2D CSP features matrix.
    """
    if len(tasks) < 2:
        print("Must have at least 2 tasks for filtering.")
        return (None,) * len(tasks)
    else:
        filters = ()
        iterator = range(0, len(tasks))
        for x in iterator:
            # Find Rx
            Rx = covarianceMatrix(tasks[x][0])
            for t in range(1, len(tasks[x])):
                Rx += covarianceMatrix(tasks[x][t])
                Rx = Rx / len(tasks[x])

            # Find not_Rx
            count = 0
            not_Rx = Rx * 0
            for not_x in [element for element in iterator if element != x]:
                for t in range(0, len(tasks[not_x])):
                    not_Rx += covarianceMatrix(tasks[not_x][t])
                    count += 1
                not_Rx = not_Rx / count

            # Find the spatial filter SFx
            SFx = spatialFilter(Rx, not_Rx)
            filters += (SFx,)

            # Special case: only two tasks, no need to compute any more mean variances
            if len(tasks) == 2:
                filters += (spatialFilter(not_Rx, Rx),)
                break
    return filters


# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    """This function computes the covariance Matrix
    Args:
        A: 2D matrix
    Returns:
        A 2D covariance matrix scaled by the variance
    """
    # Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
    Ca = np.cov(A)
    return Ca


def spatialFilter(Ra, Rb):
    R = Ra + Rb
    E, U = la.eig(R)

    # CSP requires the eigenvalues E and eigenvector U be sorted in descending order
    ord = np.argsort(E)
    ord = ord[::-1]  # argsort gives ascending order, flip to get descending
    E = E[ord]
    U = U[:, ord]

    # Find the whitening transformation matrix
    P = np.dot(np.sqrt(la.inv(np.diag(E))), np.transpose(U))

    # The mean covariance matrices may now be transformed
    Sa = np.dot(P, np.dot(Ra, np.transpose(P)))
    Sb = np.dot(P, np.dot(Rb, np.transpose(P)))

    # Find and sort the generalized eigenvalues and eigenvector
    E1, U1 = la.eig(Sa, Sb)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:, ord1]

    # The projection matrix (the spatial filter) may now be obtained
    SFa = np.dot(np.transpose(U1), P)
    # return SFa.astype(np.float32)
    return SFa


# Zero crossings: it is the number of times the waveform crosses zero. This feature provides an approximate estimation of frequency domain properties. The threshold avoids counting zero crossings induced by noise.


# def features_calculation(signal, fs, window_size, window_shift):
#     """Root Mean Square.

#     Args:
#         signal (array_like): TODO
#         fs (int): Sampling frequency
#         window_size: TODO
#         window_shift: TODO

#     Returns:
#         TODO:
#     """
#     duration = len(signal)/fs
#     n_features = int(duration/(window_size-window_shift))
#     rms_features = np.zeros(n_features)
#     mav_features = np.zeros(n_features)
#     ssi_features = np.zeros(n_features)
#     var_features = np.zeros(n_features)
#     zc_features = np.zeros(n_features)
#     wl_features = np.zeros(n_features)
#     ssc_features = np.zeros(n_features)
#     wamp_features = np.zeros(n_features)

#     for i in range(n_features):
#         idx1 = int((i*(window_size-window_shift))*fs)
#         idx2 = int(((i+1)*window_size-i*window_shift)*fs)
#         rms = np.sqrt(np.mean(np.square(signal[idx1:idx2])))
#         mav = np.mean(np.abs(signal[idx1:idx2]))
#         var = np.var(signal[idx1:idx2])
#         ssi = np.sum(np.abs(np.square(signal[idx1:idx2])))
#         wl = np.sum(np.abs(np.diff(signal[idx1:idx2])))
#         def zc(segment):
#                 nz_segment = []
#                 nz_indices = np.nonzero(segment)[0] # Finds the indices of the segment with nonzero values
#                 for i in nz_indices:
#                         nz_segment.append(segment[i]) # The new segment contains only nonzero values
#                         N = len(nz_segment)
#                         zc = 0
#                         for n in range(N-1):
#                                 if((nz_segment[n]*nz_segment[n+1]<0) and np.abs(nz_segment[n]-nz_segment[n+1]) >= 0.001):
#                                                    zc = zc + 1
#                 return zc
#         zc = zc(signal[idx1:idx2])
#         #Willison amplitude: it is the number of times that the difference of the amplitude between to adjacent data points exceed a predefined threshold. This feature provides information about the muscle contraction level.
#         def wamp(segment):

#             N = len(segment)
#             wamp = 0
#             for n in range(N-1):
#                      if np.abs(segment[n]-segment[n+1])>=50:
#                                      wamp = wamp + 1
#             return wamp
#         wamp = wamp(signal[idx1:idx2])
#         wamp_features[i] = wamp
#         wl_features[i] = wl
#         zc_features[i] = zc
#         ssi_features[i] = ssi
#         ssc_features= ssc
#         mav_features[i] = mav
#         var_features[i] = var
#         rms_features[i] = rms

#     return wl_features, mav_features, rms_features


def relevant_features(X, y, model):
    model.fit(X, y)
    relevance = model.feature_importances_
    indices = np.argsort(relevance)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], relevance[indices[f]]))


def PCA_dim_red(features, var_desired):
    """Dimensionality reduction of features using PCA.

    Args:
        features (matrix (2d np.array)): The feature matrix
        var_desired (float): desired preserved variance

    Returns:
        features with reduced dimensions
    """
    # PCA
    pca = sklearn.decomposition.PCA(n_components=features.shape[1] - 1)
    pca.fit(features)
    # print('pca.explained_variance_ratio_:\n',pca.explained_variance_ratio_)
    var_sum = pca.explained_variance_ratio_.sum()
    var = 0
    for n, v in enumerate(pca.explained_variance_ratio_):
        var += v
        if var / var_sum >= var_desired:
            features_reduced = sklearn.decomposition.PCA(n_components=n + 1).fit_transform(features)
            return features_reduced


def ploy_features(degree, X):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    return X_poly


def feature_scaling(feature_matrix, target, N_components, reductor=None, scaler=None):
    lda = LDA(n_components=N_components)
    minmax = MinMaxScaler(feature_range=(-1, 1))
    if not reductor:
        reductor = lda.fit(feature_matrix, target)
    feat_lda = reductor.transform(feature_matrix)
    if not scaler:
        scaler = minmax.fit(feat_lda)
    feat_lda_scaled = scaler.transform(feat_lda)

    return feat_lda_scaled, reductor, scaler


def convert_one_hot_encoding(classes):
    encoder = LabelBinarizer()
    transfomed_labels = encoder.fit_transform(classes)
    return transfomed_labels


def powermean(data, trial, fs, w):
    return (
        np.power(data[trial + fs * 4 + w[0] : trial + fs * 4 + w[1], 0], 2).mean(),
        np.power(data[trial + fs * 4 + w[0] : trial + fs * 4 + w[1], 1], 2).mean(),
        np.power(data[trial + fs * 4 + w[0] : trial + fs * 4 + w[1], 2], 2).mean(),
    )


def log_subBP_feature_extraction(alpha, beta, trials, fs, w):
    # number of features combined for all trials
    n_features = 15
    # initialize the feature matrix
    X = np.zeros((len(trials), n_features))

    # Extract features
    for t, trial in enumerate(trials):
        power_c31, power_c41, power_cz1 = powermean(alpha[0], trial, fs, w)
        power_c32, power_c42, power_cz2 = powermean(alpha[1], trial, fs, w)
        power_c33, power_c43, power_cz3 = powermean(alpha[2], trial, fs, w)
        power_c34, power_c44, power_cz4 = powermean(alpha[3], trial, fs, w)
        power_c31_b, power_c41_b, power_cz1_b = powermean(beta[0], trial, fs, w)

        X[t, :] = np.array(
            [
                np.log(power_c31),
                np.log(power_c41),
                np.log(power_cz1),
                np.log(power_c32),
                np.log(power_c42),
                np.log(power_cz2),
                np.log(power_c33),
                np.log(power_c43),
                np.log(power_cz3),
                np.log(power_c34),
                np.log(power_c44),
                np.log(power_cz4),
                np.log(power_c31_b),
                np.log(power_c41_b),
                np.log(power_cz1_b),
            ]
        )

    return X


def dwt_features(data, trials, level, sampling_freq, w, n, wavelet):

    # number of features per trial
    n_features = 12
    # allocate memory to store the features
    X = np.zeros((len(trials), n_features))

    # Extract Features
    for t, trial in enumerate(trials):
        signals = data[trial + sampling_freq * 4 + (w[0]) : trial + sampling_freq * 4 + (w[1])]
        coeffs_c3 = pywt.wavedec(data=signals[:, 0], wavelet=wavelet, level=level)
        coeffs_c4 = pywt.wavedec(data=signals[:, 1], wavelet=wavelet, level=level)
        coeffs_cz = pywt.wavedec(data=signals[:, 2], wavelet=wavelet, level=level)

        X[t, :] = np.array(
            [
                np.std(coeffs_c3[n]),
                np.std(coeffs_c4[n]),
                np.std(coeffs_cz[n]),
                np.sqrt(np.mean(np.square(coeffs_c3[n]))),
                np.sqrt(np.mean(np.square(coeffs_c4[n]))),
                np.sqrt(np.mean(np.square(coeffs_cz[n]))),
                np.mean(coeffs_c3[n] ** 2),
                np.mean(coeffs_c4[n] ** 2),
                np.mean(coeffs_cz[n] ** 2),
                np.mean(coeffs_c3[n]),
                np.mean(coeffs_c4[n]),
                np.mean(coeffs_cz[n]),
            ]
        )

    return X


def RF_feature_importance(X_train, y_train, X_test, y_test):

    for k_est in [10, 30, 50, 100, 1000]:
        for thres in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.15]:
            try:
                clf = RandomForestClassifier(
                    n_estimators=k_est, n_jobs=-1, warm_start=True, random_state=0
                )

                # Create a selector object that will use the random forest classifier to identify
                # features that have an importance of more than 0.15
                sfm = SelectFromModel(clf, threshold=thres)

                # Train the selector
                sfm = sfm.fit(X_train, y_train)

                # save (sfm, '/mnt/c/Users/Aleksandar/Documents/TUM/ICS Research/EMG_scripts/model/sfmRA_1s.sav')

                # Transform the data to create a new dataset containing only the most important features
                # Note: We have to apply the transform to both the training X and test X data.
                X_important_train = sfm.transform(X_train)
                X_important_test = sfm.transform(X_test)

                # Create a new random forest classifier for the most important features
                # clf_important = RandomForestClassifier(n_estimators=15, random_state=0, n_jobs=-1)

                # Train the new classifier on the new dataset containing the most important features
                RA = clf.fit(X_important_train, y_train)

                # save (RA, '/mnt/c/Users/Aleksandar/Documents/TUM/ICS Research/EMG_scripts/model/RA_1s.sav')

                y_pred = clf.predict(X_important_test)

                # print("\n\nClassification Report on Test set before the retraining:")
                # print(confusion_matrix(y_test, y_pred))
                n_correct_clf = len(np.where(y_test - y_pred == 0)[0])
                accuracy = (float(n_correct_clf) / len(y_pred)) * 100
                print("No. of estimators: %3d, Thres: %f, Accuracy: %f" % (k_est, thres, accuracy))
            except:
                pass


def extract_features(epoch, spatial_filter):
    feature_matrix = np.dot(spatial_filter, epoch)
    variance = np.var(feature_matrix, axis=1)
    if np.all(variance == 0):
        return np.zeros(spatial_filter.shape[0])
    features = np.log(variance / np.sum(variance))
    return features


def select_filters(filters, num_components=None, verbose=False):
    if verbose:
        print(
            "Incomming filters:",
            filters.shape,
            "\nNumber of components:",
            num_components if num_components else " all",
        )
    if num_components == None:
        return filters
    assert num_components <= filters.shape[0] / 2, "The requested number of components is too high"
    selection = list(range(0, num_components)) + list(
        range(filters.shape[0] - num_components, filters.shape[0])
    )
    reduced_filters = filters[selection, :]
    return reduced_filters


# Select the number of used spatial components
n_components = None
# assign None for all components


def CSP(tasks):
    """This function extracts Common Spatial Pattern (CSP) features.
    Args:
        For N tasks, N arrays are passed to CSP each with dimensionality (# of
        trials of task N) x (feature vector)
    Returns:
        A 2D CSP features matrix.
    """
    if len(tasks) < 2:
        print("Must have at least 2 tasks for filtering.")
        return (None,) * len(tasks)
    else:
        filters = ()
        iterator = range(0, len(tasks))
        for x in iterator:
            # Find Rx
            Rx = covarianceMatrix(tasks[x][0])
            for t in range(1, len(tasks[x])):
                Rx += covarianceMatrix(tasks[x][t])
                Rx = Rx / len(tasks[x])

            # Find not_Rx
            count = 0
            not_Rx = Rx * 0
            for not_x in [element for element in iterator if element != x]:
                for t in range(0, len(tasks[not_x])):
                    not_Rx += covarianceMatrix(tasks[not_x][t])
                    count += 1
                not_Rx = not_Rx / count

            # Find the spatial filter SFx
            SFx = spatialFilter(Rx, not_Rx)
            filters += (SFx,)

            # Special case: only two tasks, no need to compute any more mean variances
            if len(tasks) == 2:
                filters += (spatialFilter(not_Rx, Rx),)
                break
    return filters


# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
    """This function computes the covariance Matrix
    Args:
        A: 2D matrix
    Returns:
        A 2D covariance matrix scaled by the variance
    """
    # Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
    Ca = np.cov(A)
    return Ca


def spatialFilter(Ra, Rb):
    R = Ra + Rb
    E, U = la.eig(R)

    # CSP requires the eigenvalues E and eigenvector U be sorted in descending order
    ord = np.argsort(E)
    ord = ord[::-1]  # argsort gives ascending order, flip to get descending
    E = E[ord]
    U = U[:, ord]

    # Find the whitening transformation matrix
    P = np.dot(np.sqrt(la.inv(np.diag(E))), np.transpose(U))

    # The mean covariance matrices may now be transformed
    Sa = np.dot(P, np.dot(Ra, np.transpose(P)))
    Sb = np.dot(P, np.dot(Rb, np.transpose(P)))

    # Find and sort the generalized eigenvalues and eigenvector
    E1, U1 = la.eig(Sa, Sb)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:, ord1]

    # The projection matrix (the spatial filter) may now be obtained
    SFa = np.dot(np.transpose(U1), P)
    # return SFa.astype(np.float32)
    return SFa
