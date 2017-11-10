import numpy as np

from Classifiers.AClassifier import AClassifier


class SoftEnsemble(AClassifier):
    """
    An Ensemble that predicts classes by adding the soft predictions of
    its classifiers.
    All classifiers' predictions are weighted the same.
    """

    def __init__(self, classifiers: [AClassifier]):
        super().__init__()
        self.classifiers = classifiers

    def train(self, x: np.array, y: np.array, classes: np.array=None) -> None:
        super().train(x, y, classes)  # Set self.classes.
        for c in self.classifiers:
            c.train(x, y, classes)

    def predict(self, x: np.array) -> float:
        return self.classes[np.argmax(self.predict_soft(x), axis=1)]

    def predict_soft(self, x: np.array) -> np.array:
        prob = np.zeros((x.shape[0], len(self.classes)))  # [MxK]
        for c in self.classifiers:
            prob += c.predict_soft(x)

        # Convert to [0,1]
        return prob / np.atleast_2d(np.sum(prob, axis=1)).T
