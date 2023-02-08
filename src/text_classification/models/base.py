import numpy as np

from text_classification.data import Samples


class Model():
    """
    Abstract base class for all models.
    """
    def fit(self, X: Samples, y: np.ndarray) -> 'Model':
        """
        Given a set of samples and their labels, fit the model.

        :param X: samples
        :param y: labels
        :return: self
        """
        raise NotImplementedError()
    
    def predict(self, X: Samples) -> np.ndarray:
        """
        Given a set of samples, predict their labels.

        :param X: samples
        :return: predicted labels
        """
        raise NotImplementedError()
