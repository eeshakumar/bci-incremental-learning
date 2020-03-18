import numpy as np
from .base_classifier import Classifier, ClassificationReport
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


class SVM(Classifier):
    """
    A classifier from SciPy
    https://scikit-learn.org/stable/modules/svm.html
    """

    def __init__(self, kernel, C, gamma):
        super(SVM, self).__init__()

        self.steps = 10
        self.regularization = 0.1
        self.num_epochs = None
        self.shuffle = False
        self.dimensions = 12 # Hardcoded for the Graz Dataset

        # https://scikit-learn.org/stable/modules/svm.html#using-the-gram-matrix
        self.clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    def static_opts(ctype, **kwargs):
        return  {}

    # def run(self, X_train, Y_train, X_Test, Y_test, **kwargs):
    #     self.dimensions = X_train.shape[1]
    #     example_id = np.array(['%d' % i for i in range(len(Y_train))])
    #     x_column_name = 'x'
    #     example_id_column_name = 'example_id'
    #
    #     train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #         x={x_column_name: X_train, example_id_column_name: example_id},
    #         y=Y_train,
    #         num_epochs=self.num_epochs,
    #         shuffle=self.shuffle
    #     )
    #     self.clf = tf.contrib.learn.SVM(
    #         example_id_column=example_id_column_name,
    #         feature_columns=(tf.contrib.layers.real_valued_column(
    #             column_name=x_column_name, dimension=self.dimensions),),
    #         l2_regularization=self.regularization
    #     )
    #     self.clf.fit(input_fn=train_input_fn, steps=self.steps)
    #     self.clf.preict()

    def run(self, x_train, y_train, x_test, y_test, **kwargs):
        self.clf.fit(x_train, y_train)
        y_pred = self.clf.predict(x_test)
        classification_report = ClassificationReport(y_pred, y_test)
        classification_report.generate()
        return classification_report, self

    def opt(self, x_train, y_train):
        C_range = np.linspace(10, 10000)
        gamma_range = np.linspace(10000, 0.01)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(self.clf, param_grid=param_grid, cv=cv)
        grid.fit(x_train, y_train)

        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        predictions = self.clf.predict(x_test)
        return predictions
