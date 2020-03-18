from abc import ABC
from reporting.report import ClassificationReport


class Classifier(ABC):

    def __init__(self):
        pass

    @staticmethod
    def static_opts(ctype, **kwargs):
       return {}

    def run(self, x_train, y_train, x_test, y_test, **kwargs):
        return ClassificationReport, self

    def train(self, x_train, y_train, x_test, y_test, **kwargs):
        return self.run(x_train, y_train, x_test, y_test)

    def opt(self, x_train, y_train):
        return self
