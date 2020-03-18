from classifier.svm import SVM


def classifier_train(classifier, kernel,  x_k, y_k):
    # n_obs, n_features = x_k.shape[0], x_k.shape[1]
    classifier = SVM(kernel, C=4, gamma=4)
    classifier.fit(x_k, y_k)
    # classifier.opt(x_k, y_k)
    return classifier