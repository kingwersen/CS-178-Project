import numpy as np


class AClassifier:

    def __init__(self):
        self.alpha = 1

    def train(self, x: np.array, y: np.array) -> None:
        raise NotImplementedError()

    def predict(self, x: np.array) -> np.array:
        raise NotImplementedError()

    def error(self, x: np.array, y: np.array) -> float:
        yh = self.predict(x)
        return np.mean(y != yh)

    def auc(self, x: np.array, y: np.array, alpha: float=1) -> float:
        # TODO
        pass