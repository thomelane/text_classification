from typing import Optional
import numpy as np

from text_classification.data import Samples
from text_classification.models.base import Model


class RandomClass(Model):
    """
    A trivial baseline model that predicts a random class.
    """
    def __init__(self):
        self._classes: Optional[np.ndarray] = None

    def fit(self, _: Optional[Samples], y: np.ndarray) -> 'RandomClass':
        """
        Given just the labels, 'fit' the model by learning the possible classes.

        :param _: unused because we don't need any features for this model
        :param y: labels
        :return: self
        """
        self._classes = np.unique(y)
        return self
    
    def predict(self, X: Samples) -> np.ndarray:
        if self._classes is None:
            raise ValueError('Call `fit` before `predict`')
        return np.random.choice(self._classes, len(X))
