from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import json
from typing import List, Dict, Optional, Any, Tuple
import random

from sklearn.preprocessing import LabelEncoder


Sample = Dict[str, Optional[Any]]  # could have missing values
Samples = List[Sample]


def load_json_tar_gz_as_dict(archive_path: Path) -> Samples:
    """
    Given a path to a .tar.gz file, extracts the file, and loads the JSON file within.

    :param archive_path: path to .tar.gz file
    :return: list of samples
    """
    assert str(archive_path).endswith(".tar.gz")
    filename = str(archive_path.name).replace(".tar.gz", "")
    with TemporaryDirectory() as tmp_dir:
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp_dir)
        with open(Path(tmp_dir, filename), encoding="utf-8") as f:
            data = json.load(f)
    return data


def split_samples(samples: Samples, frac: float) -> Tuple[Samples, Samples]:
    """
    Given a list of samples, splits the samples into two groups based on the given fraction.

    :param samples: list of samples
    :param frac: fraction of samples to put in the first group
    :return:a tuple of two lists of samples
    """
    assert 0.0 < frac < 1.0
    num_samples = len(samples)
    num_samples_a = int(num_samples * frac)
    # random shuffle all indices
    idxs = list(range(num_samples))
    random.shuffle(idxs)
    # obtain indices for each group
    idxs_a = idxs[:num_samples_a]
    idxs_b = idxs[num_samples_a:]
    # use indices to split data into two groups
    samples_a = [samples[i] for i in idxs_a]
    samples_b = [samples[i] for i in idxs_b]
    return samples_a, samples_b


def add_id_to_samples(samples: Samples, id_field: str) -> Samples:
    """
    Adds an id field (based on the position of the sample in the list),
    so data lineage can be tracked.

    :param samples: list of samples
    :param id_field: name of the field to record the id
    """
    for i, sample in enumerate(samples):
        sample[id_field] = i
    return samples


def add_class_idx_to_samples(samples: Samples, label_encoder: LabelEncoder, class_field: str, class_idx_field: str) -> Samples:
    """
    Adds a class index field, so the class can be used as a label by the model.
    
    :param samples: list of samples
    :param label_encoder: label encoder
    :param class_field: name of the field containing the class
    :param class_idx_field: name of the field to record the class index
    """
    samples = samples.copy()
    for sample in samples:
        class_name = sample[class_field]
        class_idx = label_encoder.transform([class_name])[0]
        sample[class_idx_field] = class_idx
    return samples


def remove_no_input_samples(samples: Samples, fields: List[str]) -> Samples:
    """
    Removes samples that have no input (i.e. all fields are missing).
    
    :param samples: list of samples
    :param fields: list of fields to check for missing values
    """
    output_samples = []
    for sample in samples:
        missing = []
        for field in fields:
            value = sample[field]
            if isinstance(value, str):
                missing.append(len(value.strip()) == 0)
            else:
                missing.append(value is None)
        all_missing = all(missing)
        if not all_missing:
            output_samples.append(sample)
    return output_samples
