from typing import Optional
import numpy as np

from text_classification.data import Samples
from text_classification.models.base import Model


class LabelDistribution(Model):
    """
    A trivial baseline model that predicts the label distribution of the training set.
    """
    def __init__(self):
        self._label_distribution: Optional[np.ndarray] = None

    def fit(self, _: Optional[Samples], y: np.ndarray) -> 'LabelDistribution':
        """
        Given just the labels, 'fit' the model by learning the label distribution.

        :param _: unused because we don't need any features for this model
        :param y: labels
        :return: self
        """
        self._label_distribution = np.bincount(y) / len(y)
        return self
    
    def predict(self, X: Samples) -> np.ndarray:
        if self._label_distribution is None:
            raise ValueError('Call `fit` before `predict`')
        return np.random.choice(len(self._label_distribution), len(X), p=self._label_distribution)
