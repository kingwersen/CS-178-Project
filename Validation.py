import numpy as np

from Classifiers.AClassifier import AClassifier


def cross_validate(classifier: AClassifier, x: np.array, y: np.array, k: int=1) -> float:
    assert (x.shape[0] == y.shape[0])
    sum_err = 0
    for i in range(k):
        xTrain, yTrain, xValid, yValid = split_data(x, y, k, i)
        classifier.train(xTrain, yTrain)
        sum_err += classifier.error(xValid, yValid)
    return sum_err / k


def split_data(x: np.array, y: np.array, k: int, i: int) -> (np.array, np.array, np.array, np.array):
    assert(x.shape[0] == y.shape[0])
    assert(0 <= i < k)

    l = int(x.shape[0] / k)
    xt = np.concatenate((x[0:l*i, :], x[l*(i+1):l*k, :]), axis=0)
    yt = np.concatenate((y[0:l*i], y[l*(i+1):l*k]), axis=0)
    xv = x[l*i:l*(i+1), :]
    yv = y[l*i:l*(i+1)]

    return xt, yt, xv, yv


