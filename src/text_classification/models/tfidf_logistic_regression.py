from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from text_classification.data import Samples
from text_classification.models.base import Model
from text_classification.transforms import concat_fields


class TfidfLogisticRegression(Model):
    """
    A baseline model that uses TF-IDF to 'embed' the samples and then uses
    a logistic regression 'head' to predict the labels.
    """
    def __init__(
        self,
        fields: List[str],
        max_features: int = 10000,
        stop_words: str = "english"
    ):
        self._fields = fields
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words
        )
        self.logistic_regression = LogisticRegression()

    def fit(self, X: Samples, y: np.ndarray) -> 'TfidfLogisticRegression':
        X_text = [concat_fields(sample, self._fields) for sample in X]
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
        self.logistic_regression.fit(X_tfidf, y)
        return self
    
    def embed(self, X: Samples) -> np.ndarray:
        X_text = [concat_fields(sample, self._fields) for sample in X]
        X_tfidf = self.tfidf_vectorizer.transform(X_text)
        X_tfidf = X_tfidf.toarray()  # type: ignore
        return X_tfidf
    
    def predict(self, X: Samples) -> np.ndarray:
        X_text = [concat_fields(sample, self._fields) for sample in X]
        X_tfidf = self.tfidf_vectorizer.transform(X_text)
        return self.logistic_regression.predict(X_tfidf)
