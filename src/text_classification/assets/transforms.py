from dagster import asset, Output

from text_classification.data import Samples
from text_classification.transforms import LabelTransform
import text_classification.config as cfg


@asset
def label_transform(source_train: Samples) -> Output[LabelTransform]:
    """
    Converts labels to integers (so we can use them to train a model).
    Also manages the mapping between the integer and string labels.

    Applied to `source_train` instead of `filtered_train_set` because we want to
    ensure we have all the classes in this label transform. Unlikely, but it's
    possible some classes were sampled/filtered out.

    :param source_train: source_train asset
    :return: label_transform asset
    """
    transform = LabelTransform(
        class_field=cfg.CLASS_FIELD,
        class_labels=cfg.CLASS_LABELS
    )
    transform.fit(source_train)
    idx_to_label = {str(i): l for i, l in enumerate(transform.labels)}
    metadata = { "idx_to_label": idx_to_label }
    output = Output(transform, metadata=metadata)
    return output
