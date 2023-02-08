from typing import Optional, List

import numpy as np
import torch

from text_classification.models.base import Model
from text_classification.models.sentence_transformer_nn.train import train


class SentenceTransformerNN(Model):
    """
    A model that uses sentence_transformers to embed the samples and then uses
    a neural network 'head' to predict the labels.
    """
    def __init__(
        self,
        version: str,
        hidden_dims: List[int] = [128],  # type: ignore
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        max_epochs: int = 50,
        overfit_batches: float = 0.0,
        dropout: float = 0.05
    ):
        self.version = version
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.overfit_batches = overfit_batches
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.model: Optional[torch.nn.Module] = None

    def fit_from_embeddings(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray
    ):
        self.model = train(
            version=self.version,
            model_name=self.__class__.__name__,
            hidden_dims=self.hidden_dims,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            overfit_batches=self.overfit_batches,
            dropout=self.dropout,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid
        )
        return self
    
    def predict_from_embeddings(self, X_embeddings: np.ndarray) -> np.ndarray:
        torch_embeddings = torch.from_numpy(X_embeddings).float()
        assert self.model is not None
        return self.model(torch_embeddings).argmax(dim=1).numpy()
