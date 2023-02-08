from typing import Optional
import numpy as np

from text_classification.data import Samples
from text_classification.models.base import Model


class MostCommonClass(Model):
    """
    A trivial baseline model that always predicts the most common class in the training set.
    """
    def __init__(self):
        self._most_common_class: Optional[int] = None

    def fit(self, _: Optional[Samples], y: np.ndarray) -> 'MostCommonClass':
        """
        Given just the labels, 'fit' the model by finding the most common class.

        :param _: unused because we don't need any features for this model
        :param y: labels
        :return: self
        """
        self._most_common_class = int(np.bincount(y).argmax())
        return self
    
    def predict(self, X: Samples) -> np.ndarray:
        if self._most_common_class is None:
            raise ValueError('Call `fit` before `predict`')
        return np.array([self._most_common_class] * len(X))
