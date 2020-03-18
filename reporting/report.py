from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


class ClassificationReport:

    def __init__(self, pred, test):
        self.pred = pred
        self.test = test
        self.generate()
        self.accuracy = None
        self.confusion_matrix = None
        self.report = None

    def generate(self):
        self.accuracy = accuracy_score(self.test, self.pred) * 100
        self.confusion_matrix = confusion_matrix(self.test, self.pred)
        self.report = classification_report(self.test, self.pred)

    def __str__(self):
        return str(self.accuracy) + "\n" + str(self.confusion_matrix) + "\n" + str(self.report)


class IncrementalClassificationReport:

    def __init__(self, clf, classifier_count, subset_count):
        self.clf = clf
        self.classifier_count = classifier_count
        self.subset_count = subset_count

    def generate_report(self, classifier_count, subset_count):
        self.classifier_count = classifier_count
        self.subset_count = subset_count
        report_str = "Data Subset {}:" \
                     "\n\t" \
                     "Classifier Iteration {}:" \
                     "\n\t\t" \
                     "epsilon: {}" \
                     "\n\t\t" \
                     "beta: {}" \
                     "\n\t\t" \
                     "Train Accuracy: {}" \
                     "\n\t\t" \
                     "e_kt: {}" \
                     "\n\t\t" \
                     "b_kt: {}" \
                     "\n\t\t" \
                     "Test Accuracy: {}" \
                     "\n\t\t" \
                     "Error: {}" \
                     "\n" \
            .format(self.subset_count, self.classifier_count,
                    self.clf.epsilons[self.classifier_count],
                    self.clf.betas[self.classifier_count],
                    self.clf.tr_accuracy[self.classifier_count],
                    self.clf.e_kt[self.classifier_count],
                    self.clf.b_kt[self.classifier_count],
                    self.clf.accuracy[self.classifier_count],
                    self.clf.errs[self.classifier_count])
        print(report_str)


def normal_data_split_report(x_train, y_train, x_test, y_test):
    report_str = "Training data size: {}" \
                 "\nTraining labels size: {}" \
                 "\nTest data size: {}" \
                 "\nTest label size: {}"\
        .format(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(report_str)


def incremental_data_split_report(ds_features, ds_labels, x_test, y_test):
    for i in range(len(ds_labels)):
        normal_data_split_report(ds_features[i], ds_labels[i], x_test, y_test)
        print()


def normalized_data_report(data):
    report_str = "Normalized Data:" \
                 "\n\tMean: {:.3f}" \
                 "\n\tStd dev: {:.3f}" \
                 "\n\tMin: {:.3f}" \
                 "\n\tMax: {:.3f}" \
                 "\n"\
        .format(np.nanmean(data), np.nanstd(data), np.nanmin(data), np.nanmax(data))
    print(report_str)
