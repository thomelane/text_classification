from dagster import asset, Output
import numpy as np

from text_classification.data import Samples
from text_classification.models import SentenceTransformerLogisticRegression
import text_classification.config as cfg


def embed(samples: Samples) -> Output[np.ndarray]:
    """
    Uses sentence_transformers to embed the samples.

    :param samples: samples to embed
    :return: embeddings matrix
    """
    model = SentenceTransformerLogisticRegression(
        cfg.FEATURE_FIELDS,
        cfg.SENTENCE_TRANSFORMER_MODEL
    )
    embeddings = model.embed(samples)
    metadata = {
        'model': cfg.SENTENCE_TRANSFORMER_MODEL,
        'shape': str(embeddings.shape)
    }
    output = Output(embeddings, metadata=metadata)
    return output


@asset
def train_embeddings(train_set: Samples) -> Output[np.ndarray]:
    """
    Embeds the training set.
    
    :param train_set: train_set asset
    :return: train_embeddings asset
    """
    return embed(train_set)


@asset
def validation_embeddings(validation_set: Samples) -> Output[np.ndarray]:
    """
    Embeds the validation set.
    
    :param validation_set: train_set asset
    :return: validation_embeddings asset
    """
    return embed(validation_set)
