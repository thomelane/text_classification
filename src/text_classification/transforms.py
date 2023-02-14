from typing import List, Tuple, Optional, Dict

from sklearn.preprocessing import LabelEncoder
import numpy as np

from text_classification.data import Sample, Samples


def blank_to_none(sample: Sample) -> Sample:
    """
    Converts string containing only whitespace to None.

    :param sample: a sample
    :return: a modified sample
    """
    sample = sample.copy()
    for k, v in sample.items():
        if isinstance(v, str):
            v = v.strip()
            if len(v) == 0:
                v = None
            sample[k] = v
    return sample


def extract_fields(sample: Sample, fields: List[str]) -> Sample:
    """
    Given a sample, extract a subset of fields.

    :param sample: a sample
    :param fields: list of fields to extract
    :return: a modified sample
    """
    return {k: sample[k] for k in fields}


def extract_label(
    samples: Samples,
    label_transform: 'LabelTransform'
) -> Tuple[Samples, np.ndarray]:
    """
    Given a list of samples, extract the label field and convert it to a numpy array.

    :param samples: list of samples
    :param label_transform: label transform
    :return: a tuple of (list of samples without label, a numpy array of labels)
    """
    y = label_transform.transform(samples)
    # remove label field from samples
    # avoid accidentally using label field as feature
    X = [{k: s[k] for k in s.keys() if k != label_transform.class_field} for s in samples]
    return X, y


def concat_fields(sample: Sample, fields: List[str]) -> str:
    """
    Concatenate the values of a list of fields.
    Used for combining `headline` and `short_description` fields.

    :param sample: a sample
    :param fields: list of fields to concatenate
    :return: a string containing the concatenated values
    """
    return ': '.join([str(sample[k]) for k in fields])


class LabelTransform():
    """
    Used to convert the class_id (str) to a class_idx (int).
    Can also map between:
        - the class_idx (int) and class_id (str) (e.g. "A")
        - the class_idx (int) and class_label (str) (e.g. "A: Art")
    """
    def __init__(self, class_field: str, class_labels: Optional[Dict[str, str]] = None):
        """
        Creates a new instance of LabelTransform.

        :param class_field: name of the field containing the class
        :param class_labels: mapping from class id to class label
        :return: a new instance of LabelTransform
        """
        self.class_field = class_field
        self.class_labels = class_labels
        self.label_encoder = LabelEncoder()
        
    def fit(self, samples: Samples) -> 'LabelTransform':
        _class: List[str] = [str(s[self.class_field]) for s in samples]
        self.label_encoder.fit(_class)
        return self
    
    def transform(self, samples: Samples) -> np.ndarray:
        _class: List[str] = [str(s[self.class_field]) for s in samples]
        return self.label_encoder.transform(_class)

    def idx_to_id(self, id: int) -> str:
        """
        e.g. 0 -> "C"
        """
        return self.label_encoder.classes_[id]  # type: ignore
    
    def idx_to_label(self, id: int) -> str:
        """
        e.g. 0 -> "C: Crime"
        """
        id = self.label_encoder.classes_[id]  # type: ignore
        if self.class_labels is None:
            raise ValueError('class_labels mapping was not provided.')
        return self.class_labels[id]  # type: ignore
    
    @property
    def labels(self) -> List[str]:
        """
        e.g. ["C: Crime", "J: Parenting", ...]
        """
        return [self.class_labels[s] for s in self.label_encoder.classes_]  # type: ignore
