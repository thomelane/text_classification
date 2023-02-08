from typing import Dict, Any, Union
import base64

import numpy as np
from sklearn.metrics import f1_score
from imblearn.metrics import classification_report_imbalanced

from text_classification.data import Samples
from text_classification.models import Model
from text_classification.transforms import LabelTransform


def test_model(
    model: Model,
    X: Union[Samples, np.ndarray],
    y: np.ndarray,
    label_transform: LabelTransform,
    suffix: str = "",
    from_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Given a model and some test data, calculate the F1 score and a classification report.
    
    :param model: model to test
    :param X: test samples (or embeddings)
    :param y: test labels
    :param label_transform: label transform (used in the classification report)
    :param suffix: suffix to add to the metadata keys (e.g. `train` or `valid`)
    :param from_embeddings: whether to use the model's `predict_from_embeddings` method
    :return: metadata (containing the F1 score and the classification report)
    """
    if from_embeddings:
        y_pred = model.predict_from_embeddings(X)  # type: ignore
    else:
        y_pred = model.predict(X)  # type: ignore
    # calculate macro average F1 score
    metric = f1_score(y, y_pred, average='macro')
    # calculate classification_report_imbalanced
    report = classification_report_imbalanced(
        y, y_pred,
        target_names=label_transform.labels
    )
    metadata = {
        f'f1{suffix}': metric,
        f'report{suffix}': report_to_base64(str(report))
    }
    return metadata


def report_to_base64(report: str) -> str:
    """
    Create a base64 encoded string from a classification report.

    Workaround for dagit's metadata viewer.
    It removes whitespaces and `classification_report_imbalanced` uses them for alignment.

    :param report: classification report
    :return: base64 encoded string
    """
    return base64.b64encode(str(report).encode('utf-8')).decode('ascii')


def report_from_base64(base64_str: str) -> str:
    """
    Create a classification report from a base64 encoded string.

    :param base64_str: base64 encoded string
    :return: classification report
    """
    return base64.b64decode(base64_str).decode('utf-8')
