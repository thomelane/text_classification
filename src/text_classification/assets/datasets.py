import os
from pathlib import Path
from typing import Tuple
from collections import defaultdict
import random

from dagster import asset, multi_asset, AssetOut, Output

from text_classification.data import (
    Samples,
    load_json_tar_gz_as_dict, 
    split_samples,
    add_id_to_samples,
    remove_no_input_samples
)
from text_classification.transforms import extract_fields
import text_classification.config as cfg


@asset
def source_train() -> Output[Samples]:
    """
    Extracts source/train.json.tar.gz, and loads as a dict.

    :return: source_train asset
    """
    path = Path(os.environ["DATA_ROOT"]) / "source" / "train.json.tar.gz"
    data = load_json_tar_gz_as_dict(path)
    metadata = {"num_samples": len(data)}
    output = Output(data, metadata=metadata)
    return output


@asset
def source_test() -> Output[Samples]:
    """
    Extracts source/test.json.tar.gz, and loads as a dict.

    :return: source_test asset
    """
    path = Path(os.environ["DATA_ROOT"]) / "source" / "test.json.tar.gz"
    data = load_json_tar_gz_as_dict(path)
    metadata = {"num_samples": len(data)}
    output = Output(data, metadata=metadata)
    return output


@multi_asset(
    outs={
        "train_set_unfiltered": AssetOut(),
        "validation_set": AssetOut(),
    }
)
def split_train_validation_sets(source_train: Samples) -> Tuple[Output[Samples], Output[Samples]]:
    """
    Creates a training and validation set from the source training data.
        - Selects certain fields from the source data.
        - Adds an id field, so data lineage can be tracked.
        - Shuffles the samples.
        - Splits the samples into a training set and a validation set.
    
    :param source_train: source_train asset
    :return: train_set_unfiltered and validation_set assets
    """
    fields = [cfg.CLASS_FIELD] + cfg.FEATURE_FIELDS
    samples = [extract_fields(s, fields) for s in source_train]
    samples = add_id_to_samples(samples, cfg.ID_FIELD)
    train_samples, validation_samples = split_samples(samples, cfg.TRAIN_FRACTION)
    train_metadata = {"num_samples": len(train_samples)}
    train_output = Output(train_samples, "train_set_unfiltered", metadata=train_metadata)
    validation_metadata = {"num_samples": len(validation_samples)}
    validation_output = Output(validation_samples, "validation_set", metadata=validation_metadata)
    return train_output, validation_output


@asset
def test_set(source_test: Samples) -> Output[Samples]:
    """
    Creates a test set from the source test data.
        - Selects certain fields from the source data.
        - Adds an id field, so data lineage can be tracked.
    
    :param source_test: source_test asset
    :return: test_set asset
    """
    fields = [cfg.CLASS_FIELD] + cfg.FEATURE_FIELDS
    samples = [extract_fields(s, fields) for s in source_test]
    samples = add_id_to_samples(samples, cfg.ID_FIELD)
    metadata = {"num_samples": len(samples)}
    output = Output(samples, metadata=metadata)
    return output


@asset
def train_set(train_set_unfiltered: Samples) -> Output[Samples]:
    """
    Applies row level filtering to the training set.
        - Removes samples with no input data (i.e. all fields are empty)
    
    :param train_set_unfiltered: train_set_unfiltered asset
    :return: train_set asset
    """
    samples = remove_no_input_samples(train_set_unfiltered, cfg.FEATURE_FIELDS)
    metadata = {"num_samples": len(samples)}
    output = Output(samples, metadata=metadata)
    return output


@asset
def train_set_balanced(train_set: Samples) -> Output[Samples]:
    """
    Under-samples the training set to balance the classes.
    Output will contain `min_class_count` samples from each class.

    :param train_set: train_set asset
    :return: train_set_balanced asset
    """
    # shuffle the samples
    random.shuffle(train_set)
    # groupby class
    class_samples = defaultdict(list)
    for sample in train_set:
        class_samples[sample[cfg.CLASS_FIELD]].append(sample)
    min_class_count = min(len(samples) for samples in class_samples.values())
    # take min_class_count from each class
    balanced_samples = [class_samples[k][:min_class_count] for k in class_samples]
    # flatten
    balanced_samples = [sample for samples in balanced_samples for sample in samples]
    metadata = {"num_samples": len(balanced_samples)}
    return Output(balanced_samples, metadata=metadata)
