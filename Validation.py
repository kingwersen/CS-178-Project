import numpy as np

from Classifiers.AClassifier import AClassifier


def cross_validate(classifier: AClassifier, x: np.array, y: np.array, k: int=1) -> float:
    """
    Produces the mean proportion of error across all classifiers and splits.
    For every split, adds the probabilities of each class predicted by each classifier, and predicts the
    highest-probability class. Returns the mean error across all splits.
    :param classifiers: List of AClassifier's used to classify the data. MUST support predict_soft().
    :param x: [MxN] Features for each Data.
    :param y: [Mx1] Actual Classes for each Data.
    :param k: Total number of splits: [1,M].
    :return: The mean proportion of error across all classifiers and splits.
    """
    assert (x.shape[0] == y.shape[0])

    sum_err = 0
    for i in range(k):
        xTrain, yTrain, xValid, yValid = split_data(x, y, k, i)

        classifier.train(xTrain, yTrain)
        yhat = classifier.predict(xValid)
        sum_err += np.mean(y != yhat)

    return sum_err / k


def split_data(x: np.array, y: np.array, k: int, i: int) -> (np.array, np.array, np.array, np.array):
    """
    Splits the x,y data into training and validation sets.
    :param x: [MxN] Features for each Data.
    :param y: [Mx1] Actual Classes for each Data.
    :param k: Total number of splits: [1,M].
    :param i: Current split index: [0,k).
    :return: (xTrain, yTrain, xValid, yValid)
    """
    assert(x.shape[0] == y.shape[0])
    assert(0 <= i < k <= x.shape[0])

    size = int(x.shape[0] / k)
    xt = np.concatenate((x[0:size*i, :], x[size*(i+1):size*k, :]), axis=0)
    yt = np.concatenate((y[0:size*i],    y[size*(i+1):size*k]),    axis=0)
    xv = x[size*i:size*(i+1), :]
    yv = y[size*i:size*(i+1)]

    return xt, yt, xv, yv
