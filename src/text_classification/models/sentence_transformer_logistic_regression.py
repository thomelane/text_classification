
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from text_classification.data import Samples
from text_classification.models.base import Model
from text_classification.transforms import concat_fields


class SentenceTransformerLogisticRegression(Model):
    """
    A model that uses sentence_transformers to embed the samples and then uses
    logistic regression 'head' to predict the labels.

    Can `fit` and `predict` as normal,
    but can also use `embed` and then `fit_from_embeddings` and `predict_from_embeddings`.
    Useful for working with pre-computed embeddings, rather than re-computing them every time.
    """
    def __init__(
        self,
        fields: List[str],
        model_str: str,
        solver: str = "lbfgs",
        max_iter: int = 1000
    ):
        self._fields = fields
        self._model_str = model_str
        self._sentence_transformer = SentenceTransformer(model_str)
        self._logistic_regression = LogisticRegression(solver=solver, max_iter=max_iter)

    def embed(self, X: Samples) -> np.ndarray:
        X_text = [concat_fields(sample, self._fields) for sample in X]
        X_embeddings = self._sentence_transformer.encode(X_text, show_progress_bar=True)
        return np.array(X_embeddings)

    def fit(self, X: Samples, y: np.ndarray) -> 'SentenceTransformerLogisticRegression':
        X_embeddings = self.embed(X)
        self._logistic_regression.fit(X_embeddings, y)
        return self
    
    def fit_from_embeddings(self, X_embeddings: np.ndarray, y: np.ndarray) -> 'SentenceTransformerLogisticRegression':
        self._logistic_regression.fit(X_embeddings, y)
        return self
    
    def predict(self, X: Samples) -> np.ndarray:
        X_embeddings = self.embed(X)
        return self._logistic_regression.predict(X_embeddings)
    
    def predict_from_embeddings(self, X_embeddings: np.ndarray) -> np.ndarray:
        return self._logistic_regression.predict(X_embeddings)
