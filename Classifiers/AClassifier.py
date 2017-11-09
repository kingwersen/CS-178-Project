import numpy as np


class AClassifier:
    """
    Abstract Classifier Type. Supports Training and Predicting.
    """

    def __init__(self):
        self.alpha = 1
        self.classes = np.zeros(0)

    def train(self, x: np.array, y: np.array, classes: np.array=None) -> None:
        """
        Train the classifier based on a set of features and their respective classes.
        :param x: [MxN] Features for each Data for the classifier to train on.
        :param y: [Mx1] Actual Classes for the given Features/Data.
        :return: None
        """
        if classes is not None:
            self.classes = classes
        else:
            self.classes = np.unique(y)

    def predict(self, x: np.array) -> np.array:
        """
        Predicts the most likely Class for each Data.
        :param x: [MxN] Features for each Data.
        :return: [Mx1] Highest probability Class for each Data.
        """
        return self.classes[np.argmax(self.predict_soft(x), axis=1)]

    def predict_soft(self, x: np.array) -> np.array:
        """
        Returns a matrix of probabilities of each Class for each Data.
        :param x: [MxN] Features for each Data.
        :return: [MxK] Probabilities of each Class for each Data.
        """
        raise NotImplementedError()

    def error(self, x: np.array, y: np.array) -> float:
        """
        Returns the "Magnitude" of incorrect predictions.
        :param x: [MxN] Features for each Data.
        :param y: [Mx1] Actual Classes for each Data.
        :return: The "Magnitude" of incorrect predictions.
        """
        yh = self.predict(x)
        return np.mean(y != yh)

    def auc(self, x: np.array, y: np.array, alpha: float=1) -> float:
        # TODO
        pass