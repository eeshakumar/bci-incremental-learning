from sklearn.externals import joblib
import time
import os


def save_model(clf, path_dir):
    now = time.time()
    model_name = 'clf-{}'.format(now)
    joblib.dump(clf, os.path.join(path_dir, model_name))
    print("Model saved as {}".format(model_name))


def load_model(model_name, path_dir):
    clf = joblib.load(os.path.join(path_dir, model_name))
    return clf
