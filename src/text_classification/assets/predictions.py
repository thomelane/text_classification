from dagster import asset, Output
import numpy as np

from text_classification.data import Samples
from text_classification.models import Model


@asset
def predictions(
    model: Model,
    test_set: Samples
) -> Output[np.ndarray]:
    """
    Obtains predictions from a model.
    Uses the test set as an example of how to use the model to make predictions,
    but this could switched out for any other dataset.

    :param sentence_transformer_logistic_regression: model asset
    :param test_set: test_set asset
    :return: predictions asset
    """
    y_pred = model.predict(test_set)
    metadata = {
        "shape": str(y_pred.shape),
    }
    output = Output(y_pred, metadata=metadata)
    return output
