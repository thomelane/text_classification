from typing import Tuple, Dict

import numpy as np
from dagster import asset, Output, OpExecutionContext

from text_classification.data import Samples
from text_classification.models import (
    Model,
    MostCommonClass,
    LabelDistribution,
    RandomClass,
    TfidfLogisticRegression,
    SentenceTransformerLogisticRegression,
    SentenceTransformerNN
)
from text_classification.transforms import LabelTransform, extract_label
from text_classification.metrics import test_model
import text_classification.config as cfg


def train_and_test_model(
    model: Model,
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Tuple[Model, Dict[str, float]]:
    """
    Used to train and test a model.

    :param model: model to train and test
    :param train_set: training set
    :param validation_set: validation set
    :param label_transform: label transform
    :return: trained model and associated metadata
    """
    X_train, y_train = extract_label(train_set, label_transform)
    X_valid, y_valid = extract_label(validation_set, label_transform)
    model.fit(X_train, y_train)
    metadata_train = test_model(model, X_train, y_train, label_transform, suffix="_train")
    metadata_valid = test_model(model, X_valid, y_valid, label_transform, suffix="_valid")
    metadata = {**metadata_train, **metadata_valid}
    return model, metadata


def train_and_test_model_from_embeddings(
    model: Model,
    train_set: Samples,
    train_embeddings: np.ndarray,
    validation_set: Samples,
    validation_embeddings: np.ndarray,
    label_transform: LabelTransform
) -> Tuple[Model, Dict[str, float]]:
    """
    Used to train and test a model from pre-computed embeddings.

    :param model: model to train and test
    :param train_set: training set
    :param validation_set: validation set
    :param label_transform: label transform
    :return: trained model and associated metadata
    """
    _, y_train = extract_label(train_set, label_transform)
    _, y_valid = extract_label(validation_set, label_transform)
    model.fit_from_embeddings(train_embeddings, y_train)  # type: ignore
    metadata_train = test_model(model, train_embeddings, y_train, label_transform, suffix="_train", from_embeddings=True)
    metadata_valid = test_model(model, validation_embeddings, y_valid, label_transform, suffix="_valid", from_embeddings=True)
    metadata = {**metadata_train, **metadata_valid}
    return model, metadata


@asset
def most_common_class_model(
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Output[Model]:
    """
    Creates a model that always predicts the most common class.

    :param train_set: train_set asset
    :param validation_set: validation_set asset
    :param label_transform: label_transform asset
    :return: most_common_class_model asset
    """
    model = MostCommonClass()
    model, metadata = train_and_test_model(model, train_set, validation_set, label_transform)
    output = Output(model, metadata=metadata)
    return output


@asset
def label_distribution_model(
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Output[Model]:
    """
    Creates a model that makes predictions according to the label distribution.

    :param train_set: train_set asset
    :param validation_set: validation_set asset
    :param label_transform: label_transform asset
    :return: label_distribution_model asset
    """
    model = LabelDistribution()
    model, metadata = train_and_test_model(model, train_set, validation_set, label_transform)
    output = Output(model, metadata=metadata)
    return output


@asset
def random_class_model(
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Output[Model]:
    """
    Creates a model that makes random predictions.

    :param train_set: train_set asset
    :param validation_set: validation_set asset
    :param label_transform: label_transform asset
    :return: random_class_model asset
    """
    model = RandomClass()
    model, metadata = train_and_test_model(model, train_set, validation_set, label_transform)
    output = Output(model, metadata=metadata)
    return output


@asset
def tfidf_logistic_regression_model(
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Output[Model]:
    """
    Creates a model that uses TF-IDF features and a logistic regression classifier.

    :param train_set: train_set asset
    :param validation_set: validation_set asset
    :param label_transform: label_transform asset
    :return: tfidf_logistic_regression_model asset
    """
    model = TfidfLogisticRegression(cfg.FEATURE_FIELDS)
    model, metadata = train_and_test_model(model, train_set, validation_set, label_transform)
    metadata['num_features'] = len(model.tfidf_vectorizer.vocabulary_)  # type: ignore
    output = Output(model, metadata=metadata)
    return output


@asset
def sentence_transformer_logistic_regression(
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Output[Model]:
    """
    Creates a model that uses sentence transformer embeddings and a logistic regression classifier.

    :param train_embeddings: train_embeddings asset
    :param validation_embeddings: validation_embeddings asset
    :param train_set: train_set asset
    :param validation_set: validation_set asset
    :param label_transform: label_transform asset
    :return: sentence_transformer_logistic_regression asset
    """
    model = SentenceTransformerLogisticRegression(cfg.FEATURE_FIELDS, cfg.SENTENCE_TRANSFORMER_MODEL)
    model, metadata = train_and_test_model_from_embeddings(
        model,
        train_set,
        train_embeddings,
        validation_set,
        validation_embeddings,
        label_transform
    )
    output = Output(model, metadata=metadata)
    return output


@asset
def sentence_transformer_nn(
    context: OpExecutionContext,
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    train_set: Samples,
    validation_set: Samples,
    label_transform: LabelTransform
) -> Output[SentenceTransformerNN]:
    """
    Creates a model that uses sentence transformer embeddings and a neural network classifier.
    
    :param context: context of the operation (to get run_id to label the run)
    :param train_embeddings: train_embeddings asset
    :param validation_embeddings: validation_embeddings asset (also used in the training loop)
    :param train_set: train_set asset
    :param validation_set: validation_set asset (also used in the training loop)
    :param label_transform: label_transform asset
    :return: sentence_transformer_nn asset
    """
    model = SentenceTransformerNN(version=context.run_id)
    _, y_train = extract_label(train_set, label_transform)
    _, y_valid = extract_label(validation_set, label_transform)
    model.fit_from_embeddings(train_embeddings, y_train, validation_embeddings, y_valid)  # type: ignore
    metadata_train = test_model(model, train_embeddings, y_train, label_transform, suffix="_train", from_embeddings=True)
    metadata_valid = test_model(model, validation_embeddings, y_valid, label_transform, suffix="_valid", from_embeddings=True)
    metadata = {**metadata_train, **metadata_valid}
    output = Output(model, metadata=metadata)
    return output


@asset
def model(
    sentence_transformer_logistic_regression: Model,
    test_set: Samples,
    label_transform: LabelTransform
) -> Output:
    """
    Sets the model to the best model found during the training phase.
    And computes the final test metrics.

    :param sentence_transformer_logistic_regression: model asset
    :param test_set: test_set asset
    :param label_transform: label_transform asset
    :return: test_metrics asset
    """
    model = sentence_transformer_logistic_regression
    X_test, y_test = extract_label(test_set, label_transform)
    metadata = test_model(
        model,
        X_test,
        y_test,
        label_transform,
        suffix="_test"
    )
    output = Output(model, metadata=metadata)
    return output
