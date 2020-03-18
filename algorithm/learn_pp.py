import numpy as np
from .train import classifier_train
from .test import classifier_test
from sklearn.metrics import accuracy_score
from reporting.report import IncrementalClassificationReport


def weighted_majority(clf, x_k, y_k, c_count):
    n_experts = c_count
    w = np.log(1.0 / clf.betas[0:n_experts])
    p = np.zeros((len(y_k), clf.no_of_classes))
    for k in range(0, n_experts):
        y = classifier_test(clf.classifiers[k], x_k)
        for i in range(0, len(y)):
            p[i, y[i]] = p[i, y[i]] + w[k]
    # return indices of classes with max probability
    predictions = np.argmax(p, axis=1)
    return predictions


def learn(clf, kernel, ds_features, ds_labels, x_test, y_test, sample_ratio=1):
    inc_clf_report = IncrementalClassificationReport(clf, 0, 0)
    Tk = clf.iterations
    K = len(ds_labels)
    clf.classifiers = [None] * (Tk * K)
    clf.betas = np.zeros((Tk * K, 1))
    clf.errs = np.zeros((Tk * K, 1))
    clf.tr_accuracy = np.zeros((Tk * K, 1))
    clf.accuracy = np.zeros((Tk * K, 1))
    clf.epsilons = np.zeros((Tk * K, 1))
    clf.b_kt = np.zeros((Tk * K, 1))
    clf.e_kt = np.zeros((Tk * K, 1))

    c_count = 0
    for k in range(0, K):
        x_k = ds_features[k]
        y_k = ds_labels[k]

        # Distribution D for weights, uniform distribution
        D = np.ones((len(y_k), 1))
        D = D / len(y_k)

        # Use prior information for subsequent classifiers
        if k > 0:
            predictions = weighted_majority(clf, x_k, y_k, c_count)
            epsilon_kt = np.sum(D[np.where(predictions != y_k)])
            clf.epsilons[c_count] = epsilon_kt
            beta_kt = epsilon_kt / (1 - epsilon_kt)
            clf.betas[c_count] = beta_kt
            D[np.where(predictions == y_k)] = beta_kt * D[np.where(predictions == y_k)]

        for t in range(0, Tk):

            # 1. Normalise distribution of feature weights
            D = D / np.sum(D)

            while True:
                # 2. Sample random indices from distribution D as training and test data.
                i = np.random.choice(range(len(D)), int(sample_ratio * len(D)), False, list(D.flatten()))
                tr_xt = np.take(x_k, i, axis=0)
                tr_yt = np.take(y_k, i)

                # 3. Train model depending on sampled indices
                clf.classifiers[c_count] = classifier_train(clf.base_classifier, kernel, tr_xt, tr_yt)

                # 4. Test trained model to update error epsilon_kt
                y = classifier_test(clf.classifiers[c_count], x_k)
                clf.tr_accuracy[c_count] = accuracy_score(y, y_k) * 100
                epsilon_kt = np.sum(D[np.where(y != y_k)])
                clf.epsilons[c_count] = epsilon_kt
                if epsilon_kt > 0.5 and t > 0:
                    t = t - 1
                    clf.classifiers[c_count] = None
                else:
                    clf.betas[c_count] = epsilon_kt / (1 - epsilon_kt)
                    # 5 Calculate final hypothesis as a weighted majority
                    predictions = weighted_majority(clf, x_k, y_k, c_count)
                    e_kt = np.sum(D[np.where(predictions != y_k)])
                    clf.e_kt[c_count] = e_kt
                    if e_kt > 0.5 and t > 0:
                        t = t - 1
                        clf.classifiers[c_count] = None
                    else:
                        break

            # 6. Update normalised error on final Hypothesis to update weights distribution
            b_kt = e_kt / (1 - e_kt)
            clf.b_kt[c_count] = b_kt
            D[np.where(predictions == y_k)] = b_kt * D[np.where(predictions == y_k)]

            # Predict on unseen test data
            predictions = weighted_majority(clf, x_test, y_test, c_count)
            te_accuracy = accuracy_score(predictions, y_test) * 100
            clf.accuracy[c_count] = te_accuracy
            clf.errs[c_count] = np.count_nonzero(np.where(predictions != y_test)) / len(y_test)
            inc_clf_report.generate_report(c_count, k)
            c_count += 1
