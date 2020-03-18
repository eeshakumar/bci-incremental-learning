from plotting.plot import plot_xy


class IncrementalClassifier:

    def __init__(self, base_classifier, iterations, no_of_classes, classifiers=[],
                 accuracy=[], errs=[], epsilons=[], e_kt=[], b_kt=[], tr_accuracy=[], betas=0):
        self.base_classifier = base_classifier
        self.iterations = iterations
        self.no_of_classes = no_of_classes
        self.classifiers = classifiers
        self.errs = errs
        self.betas = betas
        self.accuracy = accuracy
        self.tr_accuracy = tr_accuracy
        self.epsilons = epsilons
        self.e_kt = e_kt
        self.b_kt = b_kt

    def plot_errs(self):
        plot_xy(self.errs, "Errors")

    def plot_betas(self):
        plot_xy(self.betas, "Normalized errors")

    def plot_train_accuracy(self):
        plot_xy(self.tr_accuracy, "Train Accuracies")

    def plot_test_accuracy(self):
        plot_xy(self.accuracy, "Test Accuracies")